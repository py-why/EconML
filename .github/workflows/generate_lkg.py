# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import argparse
import re
from collections import defaultdict, namedtuple
from os import listdir, path

import packaging.version
from packaging.version import Version

# We have a list of requirements files, one per python version and OS.
# We want to generate a single requirements file that specifies the requirements
# for each package contained in any of those files, along with the constraints on python version
# and OS that apply to each package.

Combo = namedtuple('Combo', ['os', 'py_version'])

# For each version of a package (say numpy==0.24.1), we'll have a set of os/py_version combos
# where it was installed; the correct constraint will be the union of all these pairs.
# However, for readability we'd like to simplify that when possible to something more readable.
# For example, if numpy==0.24.1 is installed on all versions of python and all OSes, we can just say
# "numpy==0.24.1"; if it's installed on all versions of python on ubuntu, we can say
# "numpy==0.24.1; platform_system=='Linux'".


# We'll precompute a dictionary of simple constraints, mapping from the sets of combos to a string representation
# of the constraint.
# For simplicity, we won't consider all possible constraints, just some easy to generate ones.
# In the most general case we'll OR together constraints grouped by os
def simple_constraint_map(all_combos: frozenset[Combo]) -> tuple[dict[frozenset[Combo], str],
                                                                 dict[tuple[str, frozenset[Version]], str]]:
    """
    Represent simple constraints via dictionaries.

    Parameters
    ----------
    all_combos : frozenset[Combo]
        All of the possible os/py_version pairs

    Returns
    -------
    (d1, d2): tuple[dict[frozenset[Combo], str], dict[tuple[str, frozenset[Version]], str]]
        A tuple of two dictionaries.
        The first dictionary maps from a constrained set of os/py_version pairs to the string representation of the
        constraint.
        The second dictionary maps from a tuple of (os, set of py_versions) to the string representation of the
        constraint.
    """
    all_os = frozenset({combo.os for combo in all_combos})
    all_py_versions = frozenset({combo.py_version for combo in all_combos})

    # constraint_map will map from sets of os/py_version pairs to a string representation of the constraint
    # that would restrict all possible combos to just that set;
    # we'll look up the sets of os/py_version pairs that a package is installed on and use this map to generate
    # the correct constraint for that package.
    constraint_map = {}

    # first generate simple os constraints, like "platform_system=='Linux'" or "platform_system!='Linux'"
    for os in all_os:
        # Get the set of all os/py_version pairs where the os is the given os and the py_version is anything
        filtered_combos = frozenset({combo for combo in all_combos if combo.os == os})
        constraint_map[filtered_combos] = f"; platform_system=='{os}'"
        constraint_map[all_combos - filtered_combos] = f"; platform_system!='{os}'"

    # now generate simple python version constraints,
    # like "python_version=='3.8'"", "python_version!='3.8'"; "python_version<'3.8'", "python_version>'3.8'"
    for i, py_version in enumerate(sorted(all_py_versions)):
        # Get the set of all os/py_version pairs where the py_version is the given py_version and the os is anything
        filtered_combos = frozenset({combo for combo in all_combos if combo.py_version == py_version})
        constraint_map[filtered_combos] = f"; python_version=='{py_version}'"
        constraint_map[all_combos - filtered_combos] = f"; python_version!='{py_version}'"

        if i > 0:
            less_than = frozenset({combo for combo in all_combos if combo.py_version < py_version})
            constraint_map[less_than] = f"; python_version<'{py_version}'"
        # We want to use >= next version instead of > this version
        #       because otherwise we have pairs like
        #           somelib==1.2, python_version<'3.9'
        #           somelib==1.3, python_version>'3.8'
        #       which is correct but looks more confusing than
        #           somelib==1.2, python_version<'3.9'
        #           somelib==1.3, python_version>='3.9'
        if i < len(all_py_versions)-2:
            next_version = sorted(all_py_versions)[i+1]
            greater_than = frozenset({combo for combo in all_combos if combo.py_version >= next_version})
            constraint_map[greater_than] = f"; python_version>='{next_version}'"

    # if every combination is present, we don't need to add any constraint
    constraint_map[all_combos] = ""

    # generate simple per-os python version constraints
    # we include the os in the key because we might not have every combination for every os
    # (e.g. maybe macos doesn't support python 3.8, in which case there won't be a combo for that, but there might
    # be a combo for ubuntu with python 3.8; then if we see all versions of python 3.9 and up on macos, we don't need
    # any python version constraint, whereas if we see all versions of python 3.9 and up on ubuntu,
    # we still do need a constraint since 3.8 is missing")
    os_map = {}
    for os in all_os:
        for i, py_version in enumerate(all_py_versions):
            filtered_combos = frozenset({combo for combo in all_combos
                                         if combo.os == os and combo.py_version == py_version})
            os_map[(os, frozenset({py_version}))] = f"python_version=='{py_version}'"
            if i > 0 and i < len(all_py_versions)-1:
                os_map[(os, all_py_versions - frozenset({py_version}))] = f"python_version!='{py_version}'"

            if i > 0:
                os_map[(os, frozenset({py for py in all_py_versions
                                       if py < py_version}))] = f"python_version<'{py_version}'"
            if i < len(all_py_versions)-1:
                os_map[(os, frozenset({py for py in all_py_versions
                                       if py > py_version}))] = f"python_version>'{py_version}'"

        # if every combination is present, we don't need to add any constraint for that os
        os_map[(os, all_py_versions)] = ""

    return constraint_map, os_map


# Convert between GitHub Actions' platform names and Python's platform.system() names
platform_map = {'macos': 'Darwin', 'ubuntu': 'Linux', 'windows': 'Windows'}


def make_req_file(requirements_directory, regex):
    """
    Make a unified requirements file from a directory of requirements files.

    Parameters
    ----------
    requirements_directory : str
        Directory containing requirements files

    regex : str
        Regex to match requirements file names, must have named groups "os" and "pyversion"
    """
    req_regex = r'^(?P<pkg>.*?)==(?P<version>.*)$'  # parses requirements from pip freeze results
    files = listdir(requirements_directory)

    all_combos = set()

    # We'll store the requirements for each version of each package in a dictionary
    # (e.g. "numpy" -> {0.24.1 -> {Combo1, Combo2, ...}, 0.24.2 -> {Combo3, Combo4, ...}, ...})
    # each entry of the inner dictionary will become a line in the requirements file
    # (e.g. "numpy==0.24.1; platform_system=='Linux' and python_version=='3.8' or ...")
    req_dict = defaultdict(lambda: defaultdict(set))  # package -> package_version -> set of Combos

    for file in files:
        match = re.match(regex, file)
        if not match:
            print(f"Skipping {file} because it doesn't match the regex")
            continue
        os = platform_map[match.group('os')]
        py_version = packaging.version.parse(match.group('pyversion'))
        combo = Combo(os, py_version)
        all_combos.add(combo)

        # read each line of the file
        with open(path.join(requirements_directory, file)) as lines:
            for line in lines:
                match = re.search(req_regex, line)
                pkg_version = packaging.version.parse(match.group('version'))
                req_dict[match.group('pkg')][pkg_version].add(combo)

    constraint_map, os_map = simple_constraint_map(frozenset(all_combos))
    # list of all requirements, sorted by package name and version
    reqs = []
    for pkg, versions in sorted(req_dict.items()):
        for version, combos in sorted(versions.items()):
            combos = frozenset(combos)
            req = f"{pkg}=={version}"

            if combos in constraint_map:
                suffix = constraint_map[combos]

            else:
                # we don't have a simple constraint for this package, so we need to generate a more complex one
                # which will generally be of the form:
                # "(platform_system=='os1' and (python_version=='py1' or python_version=='py2') or ...) or
                #  (platform_system=='os2' and (python_version=='py3' or ...) ..."
                #
                # that is, we will OR together constraints grouped by os
                # for some oses, we might find a nice representation for their python version constraints in the os_map
                # (e.g. "python_version=='3.8'", or "python_version<'3.8'"), in which case we'll use that;
                # for others, we'll have to OR together all of the relevant individual versions
                os_constraints = []

                os_versions = defaultdict(set)  # dictionary from os to set of python versions
                for combo in combos:
                    os_versions[combo.os].add(combo.py_version)

                # for each os, generate the corresponding constraint
                for os in sorted(os_versions.keys()):
                    versions = os_versions[os]
                    os_key = (os, frozenset(os_versions[os]))
                    if os_key in os_map:
                        constraint = os_map[os_key]
                        if constraint == "":
                            os_constraints.append(f"platform_system=='{os}'")
                        else:
                            os_constraints.append(f"platform_system=='{os}' and {constraint}")
                    else:
                        version_constraint = " or ".join([f"python_version=='{py_version}'"
                                                          for py_version in sorted(os_versions[os])])
                        os_constraints.append(f"platform_system=='{os}' and ({version_constraint})")
                if len(os_constraints) == 1:  # just one os with correspondig python versions, can use it directly
                    suffix = f"; {os_constraints[0]}"
                else:  # need to OR them together
                    suffix = f"; ({') or ('.join(os_constraints)})"

            reqs.append(f"{req}{suffix}")

    return '\n'.join(reqs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate requirements files for CI')
    parser.add_argument('requirements_directory', type=str, help='Directory containing requirements files')
    parser.add_argument('regex', type=str,
                        help='Regex to match requirements file names, must have named groups "os" and "pyversion"')
    parser.add_argument('output_name', type=str, help='File to write requirements to')
    args = parser.parse_args()

    reqs = make_req_file(args.requirements_directory, args.regex)
    with open(args.output_name, 'w') as f:
        f.write(reqs)
