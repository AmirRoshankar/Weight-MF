# Public-release audit

The public tree contains experiment code, configurations, numerical results, generated figures and tests. It excludes local run directories, source corpora, model binaries, operational logs, private working notes and credential files.

Before publication, all reachable objects in the previous Git history were scanned with Gitleaks 8.30.1 and a separate pattern-based audit that also examined commit metadata and local filesystem paths. Gitleaks reported no credential findings. The metadata audit identified private email addresses and machine-local paths; the cleaned tree removes those values, and the replacement history uses a GitHub noreply identity.

The replacement history and a fresh clone are scanned again before release. Validation also covers Python imports, objective gradients, model equivalence, dataset labels, figure regeneration, and a short CUDA control experiment. Full historical training is not rerun as part of repository packaging.

These checks cover the published Git references and file contents. Replacing branch history cannot revoke copies already downloaded or guarantee deletion of GitHub's cached/unreachable objects. A credential exposed in a previous public version would require revocation even after history removal. No matching credential was found in this audit.

For independent verification, run Gitleaks over all history and inspect local-path/email metadata separately:

    gitleaks git . --log-opts=--all --redact
    git log --all --format=fuller

Do not commit artifacts/, data/, runs/, environment files, or local credentials.

A post-publication API check confirmed that a previous, now-unreferenced commit is still retrievable by its identifier on GitHub. The fresh clone and all published branch references contain only the sanitized history. Removing GitHub-retained views/objects requires a host-side purge; branch rewriting alone does not complete that erasure.
