#!/usr/bin/env bash
#
# push.sh: verify, then commit and push the current state of the repo.
#
# Usage:
#   ./scripts/push.sh "fix: describe the change"
#   ./scripts/push.sh                # uses a default commit message
#
# Fails closed: it refuses to commit or push unless the test suite passes AND
# the experiment runs end to end. Run it from anywhere inside the repo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Prefer the project virtualenv if present. Override with PYTHON=... if desired.
if [ -z "${PYTHON:-}" ] && [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi
MSG="${1:-chore: update doctor-visit-prediction-random-vs-grouped-cross-validation}"

echo "==> Running test suite"
"$PYTHON" -m pytest -q

echo "==> Running the experiment end to end"
"$PYTHON" run_experiment.py

if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "==> No changes to commit. Pushing any unpushed commits."
else
    echo "==> Staging and committing changes"
    git add -A
    git commit -m "$MSG"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
    echo "ERROR: no 'origin' remote configured." >&2
    echo "Set it once with:" >&2
    echo "  git remote add origin git@github.com:DefinitelyScott/doctor-visit-prediction-random-vs-grouped-cross-validation.git" >&2
    exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "==> Pushing $BRANCH to origin"
git push -u origin "$BRANCH"
echo "==> Done."
