# Releasing Purrcept Core

Publishing a GitHub Release runs `.github/workflows/publish.yml`. Pushing a commit
or tag alone does not publish to PyPI. The tag must exactly match `v` followed by
`project.version` in `pyproject.toml`; this release is `v0.5.1`.

## One-time setup

On the existing [PyPI project publishing settings](https://pypi.org/manage/project/purrcept-core/settings/publishing/),
add a GitHub Trusted Publisher with these exact values:

| Field | Value |
| --- | --- |
| Owner | `Windpicker-owo` |
| Repository | `purrcept_core` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

The repository must have a GitHub environment named `pypi`. No PyPI API token or
GitHub repository secret is needed. Authentication follows the
[PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Release procedure

1. Update the package version, lockfile, README and changelog, then commit and push.
2. Check CI, create the corresponding tag on that commit, and publish a GitHub Release.
3. The workflow checks the tag/version match, resolves public dependencies, runs lint,
   formatting, strict typing and tests with the existing 100% coverage gate, then
   builds an sdist and a wheel from that sdist. Twine validates both distributions.
4. A separate `pypi` job downloads the validated artifacts and publishes through OIDC.
   This job has publishing identity but does not check out or execute project source.
5. Verify both the workflow conclusion and the new version on PyPI before announcing
   completion or publishing adapters that depend on this Core version.

Release tags are treated as immutable. Do not move a tag to retry a changed build:
fix the problem and use a new version. If only Trusted Publisher configuration or a
transient service error prevented the upload, rerun the failed job in GitHub Actions.
Inspect PyPI first if the result is uncertain: published files cannot be overwritten.
Publishing a prerelease on GitHub also triggers this workflow; use a PEP 440 prerelease
version in the package metadata when that is intended.
