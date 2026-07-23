# Security Policy

This is a personal research/portfolio project, not a maintained production
system, but reports are still welcome.

## Reporting a vulnerability

If you find a security issue in this repository (in the detection code,
the API service, the Docker image, or a dependency this project pins to a
version with a known vulnerability that hasn't been updated yet), please
open a GitHub issue on this repository with:

- A description of the issue and where it is (file/endpoint/dependency)
- Steps to reproduce, if applicable
- Why you believe it's a security issue rather than a regular bug

There is no bug bounty and no formal SLA -- this is maintained by one
person in their spare time -- but reports will be read and, if valid,
addressed.

## Scope

In scope: the API service (`src/phishnet/api.py`), the model
artifact/loading code, the Dockerfile, and this repository's pinned
dependencies.

Out of scope: the PhiUSIIL dataset itself (report data issues to the
original authors), and anything about how you've deployed this project
outside of what's documented here (e.g., if you expose the API publicly
without the rate limiting your own deployment needs, that's a
configuration choice on your end, not a vulnerability in this repo).

## What this project already does defensively

See the README's Security section and `docs/threat_model.md` for the
full picture, but briefly: the `/predict` endpoint never fetches a
user-submitted URL server-side (deliberately, to avoid making this
service an SSRF vector), strict Pydantic input validation with a
length-capped URL, rate limiting, no raw URL logging, a non-root
container user, and CI checks (`pip-audit`, `bandit`) on every push.
