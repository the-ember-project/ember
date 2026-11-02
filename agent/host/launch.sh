#!/usr/bin/env bash
# Entry point for running the agent on your own Linux machine instead of GitHub Actions.
# cron calls this once a day (see install.sh). It first updates the agent's code to the
# latest approved version on GitHub, then runs that version's job.sh, so merged pull
# requests (including changes to these scripts) take effect on the next run.
#
# Usage: launch.sh session|lab|review

main() {
    set -euo pipefail
    local mode="${1:?usage: launch.sh session|lab|review}"
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    cd "$REPO_DIR"
    export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

    # Settings and secrets, readable only by you (see install.sh).
    set -a
    # shellcheck disable=SC1090
    source "${AGIEXP_CONFIG:-$HOME/.config/agiexp/env}"
    set +a
    export AGIEXP_BRANCH="${AGIEXP_BRANCH:-main}"
    export AGIEXP_REMOTE="${AGIEXP_REMOTE:-https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git}"

    # One run at a time: a lab run and a session never overlap.
    exec 9>"$REPO_DIR/.git/agiexp.lock"
    flock 9

    echo "=== $(date -u +'%Y-%m-%d %H:%M') UTC: $mode"
    # Throw away any local change to the protected files, then take the latest approved code.
    git checkout -q HEAD -- agent .github requirements.txt
    git -c core.hooksPath=/dev/null pull -q --rebase --autostash "$AGIEXP_REMOTE" "$AGIEXP_BRANCH"

    exec bash agent/host/job.sh "$mode"
}

# Wrapped in a function so bash reads the whole script before running it, even if
# 'git pull' replaces this file mid-run.
main "$@"
exit
