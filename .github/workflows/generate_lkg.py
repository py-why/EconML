# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import re
from collections import defaultdict, namedtuple
from pkg_resources import packaging
from pkg_resources.extern.packaging.version import Version
from dataclasses import dataclass

FileParts = namedtuple('FileParts', ['os', 'py_version', 'type'])


@dataclass
class CollectionMetadata:
    all_file_parts: set[FileParts]  # all individual file parts
    py_version_oses: dict[Version, str]  # map from python version to set of oses used
    all_py_versions: set[Version]  # set of all python versions used
    reqs: dict[str, dict[str, set[FileParts]]]  # req -> pkg version -> list of file parts


# the constraint itself, plus whether it needs to be parenthesized when ANDed with other constraints
constraint_string = namedtuple('constraint_string', ['constraint', 'needs_paren'])


def constraint_to_string(cstr, in_and):
    return f"({cstr.constraint})" if cstr.needs_paren and in_and else cstr.constraint


def version_range(versions, all_py_versions):
    sorted_versions = sorted(all_py_versions)
    min_version = sorted_versions[0]
    max_version = sorted_versions[-1]
    version_inds = {v: i for i, v in enumerate(sorted_versions)}

    inds = [version_inds[v] for v in versions]
    assert (sorted(inds) == list(range(min(inds), max(inds) + 1)))  # make sure versions are contiguous
    if min_version in versions:  # no lower bound needed
        if max_version in versions:  # no upper bound needed either
            return None
        else:
            return constraint_string(f"python_version<='{max(versions)}'", False)
    elif max_version in versions:  # no upper bound needed
        return constraint_string(f"'{min(versions)}'<=python_version", False)
    elif len(versions) == 1:
        return constraint_string(f"python_version=='{versions[0]}'", False)
    else:
        return constraint_string(f"'{min(versions)}'<=python_version and python_version<='{max(versions)}'", False)


platform_map = {'macos': 'Darwin', 'ubuntu': 'Linux', 'windows': 'Windows'}


def os_constraint(oses, all_oses):
    if len(oses) == len(all_oses):  # don't need to constrain
        return None
    else:
        constraints = [f"platform_system=='{platform_map[os]}'" for os in oses]
        if len(oses) == 1:
            return constraint_string(constraints[0], False)
        else:
            return constraint_string(f"({' or '.join(constraints)})", True)


def combined_constraint(oses, versions):
    if oses is None:
        if versions is None:
            return None
        else:
            return constraint_to_string(versions, False)
    else:
        if versions is None:
            return constraint_to_string(oses, False)
        else:
            return f"{constraint_to_string(versions, True)} and {constraint_to_string(oses, True)}"


def get_reqs(metadata: CollectionMetadata):
    all_constraints = []

    reqs = metadata.reqs
    py_version_oses = metadata.py_version_oses

    for req in reqs:
        reverse_map = defaultdict(set)  # (python version, os) -> pkg version set
        for pkg_version in reqs[req]:
            for file_parts in reqs[req][pkg_version]:
                reverse_map[(file_parts.py_version, file_parts.os)].add(pkg_version)

        if not all([len(reverse_map[k]) == 1 for k in reverse_map]):
            # warn about multiple package versions
            from warnings import warn
            for k in reverse_map:
                if len(reverse_map[k]) > 1:
                    warn(f"Multiple package versions for {req}: {(k, reverse_map[k])}; defaulting to lowest version")
                    min_ver = min(reverse_map[k], key=packaging.version.parse)
                    for pkg_version in reqs[req]:
                        for file_parts in list(reqs[req][pkg_version]):
                            if file_parts.os == k[1] and pkg_version != min_ver:
                                reqs[req][pkg_version].remove(file_parts)
                            

        for pkg_version in reqs[req]:
            # break down first by python version
            py_version_map=defaultdict(set)  # python version -> os set
            for file_parts in reqs[req][pkg_version]:
                py_version_map[file_parts.py_version].add(file_parts.os)

            os_constraints=defaultdict(list)  # os constraint string -> python version list
            for py_version in py_version_map:
                os_constraints[os_constraint(py_version_map[py_version],
                                             py_version_oses[py_version])].append(py_version)

            for os_constraint_str in os_constraints:
                py_version_constraint_str=version_range(os_constraints[os_constraint_str], metadata.all_py_versions)
                combined_constraint_str=combined_constraint(os_constraint_str, py_version_constraint_str)
                all_constraints.append(
                    f"{req}=={pkg_version}; {combined_constraint_str}" if combined_constraint_str
                    else f"{req}=={pkg_version}")

    req_str='\n'.join(sorted(all_constraints))
    return req_str


def make_req_files(requirements_directory):

    files=os.listdir(requirements_directory)

    test_metadata=CollectionMetadata(set(), defaultdict(set), set(), defaultdict(lambda: defaultdict(set)))
    notebook_metadata=CollectionMetadata(set(), defaultdict(set), set(), defaultdict(lambda: defaultdict(set)))

    for file in files:
        # read each line of the file
        for line in open(os.path.join(requirements_directory, file), 'r'):
            # Regex to match requirements file names as stored by ci.yml
            file_regex=r'''^((tests-(macos|ubuntu|windows)-latest-(3\.\d+)-([^-]+)) # test filename
                              |(notebooks-(.*)-(3\.\d+)))-requirements.txt$'''        # notebook filename
            req_regex=r'^(.*?)==(.*)$'  # parses requirements from pip freeze results
            match=re.search(req_regex, line)
            file_match=re.search(file_regex, file, flags=re.VERBOSE)
            if file_match.group(2):  # we matched the test file pattern
                file_parts=FileParts(os=file_match.group(3),
                                       py_version=packaging.version.parse(file_match.group(4)),
                                       type=file_match.group(5))
                test_metadata.all_file_parts.add(file_parts)
                test_metadata.all_py_versions.add(file_parts.py_version)
                test_metadata.py_version_oses[file_parts.py_version].add(file_parts.os)
                test_metadata.reqs[match.group(1)][match.group(2)].add(file_parts)
            elif file_match.group(6):
                file_parts=FileParts(os='ubuntu',
                                       py_version=packaging.version.parse(file_match.group(8)),
                                       type=file_match.group(7))
                notebook_metadata.all_file_parts.add(file_parts)
                notebook_metadata.all_py_versions.add(file_parts.py_version)
                notebook_metadata.py_version_oses[file_parts.py_version].add(file_parts.os)
                notebook_metadata.reqs[match.group(1)][match.group(2)].add(file_parts)

    return get_reqs(test_metadata), get_reqs(notebook_metadata)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Generate requirements files for CI')
    parser.add_argument('requirements_directory', type=str, help='Directory containing requirements files')
    parser.add_argument('output_directory', type=str, help='Directory to write requirements files to')
    args=parser.parse_args()
    test_reqs, notebook_reqs=make_req_files(args.requirements_directory)
    with open(os.path.join(args.output_directory, 'lkg.txt'), 'w') as f:
        f.write(test_reqs)
    with open(os.path.join(args.output_directory, 'lkg-notebook.txt'), 'w') as f:
        f.write(notebook_reqs)
