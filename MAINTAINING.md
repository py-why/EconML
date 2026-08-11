# Maintaining EconML

Tasks that require maintainer permissions. Day-to-day contributor workflow --
environment setup, running the tests, building the docs -- is in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Release process

We use GitHub Actions to build and publish the package and documentation.  To create a new release, an admin should perform the following steps:

1. Update the version number in `econml/_version.py` and add a mention of the new version in the news section of `README.md` and commit the changes.
2. Manually run the publish_package.yml workflow to build and publish the package to PyPI.
3. Manually run the publish_docs.yml workflow to build and publish the documentation.
4. Under https://github.com/py-why/EconML/releases, create a new release with a corresponding tag, and update the release notes.
