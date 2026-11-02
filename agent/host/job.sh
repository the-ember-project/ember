#!/usr/bin/env bash
# One research session or lab run on your own Linux machine. Does the same steps as the
# GitHub workflows (.github/workflows/research.yml and lab.yml). Called by launch.sh.
#
# Usage: job.sh session|lab|review

main() {
    set -euo pipefail
    MODE="$1"
    cd "$(dirname "${BASH_SOURCE[0]}")/../.."

    # The agent's commands run as this separate user, so they can't read your secrets or
    # touch the agent's code. Leave it as 'researcher' except when testing.
    export AGENT_RUN_AS="${AGENT_RUN_AS-researcher}"
    export AGENT_CACHE_DIR="${AGENT_CACHE_DIR:-/var/cache/agiexp}"

    if [ "$MODE" = lab ] && ! grep -qvE '^\s*(#|$)' self/lab_queue.txt 2>/dev/null; then
        echo "Lab queue is empty."
        return 0
    fi

    # Python environment, kept between runs and updated when requirements.txt changes.
    local venv="${AGIEXP_VENV:-$HOME/.local/share/agiexp/venv}"
    [ -x "$venv/bin/python" ] || python3 -m venv "$venv"
    "$venv/bin/pip" install -q -r requirements.txt

    mkdir -p self knowledge research experiments "$AGENT_CACHE_DIR"
    # Folders the agent's commands may write. Files the agent created are already writable.
    chmod -R a+rwX self knowledge research experiments "$AGENT_CACHE_DIR" 2>/dev/null || true

    local status=0
    case "$MODE" in
        session) "$venv/bin/python" agent/run.py || status=$? ;;
        lab) "$venv/bin/python" agent/lab.py || status=$? ;;
        review) "$venv/bin/python" agent/review.py || status=$? ;;
        *) echo "usage: job.sh session|lab|review" >&2; return 2 ;;
    esac

    save
    return "$status"
}

save() {
    # Stop anything the agent left running in the background.
    if [ -n "$AGENT_RUN_AS" ]; then
        sudo -n -u "$AGENT_RUN_AS" pkill -KILL -u "$AGENT_RUN_AS" || true
    fi

    # Undo changes to protected files. The lab only saves experiments and its queue and log;
    # the weekly review only saves the review and the spend.
    local protected=(agent .github requirements.txt reviews)
    case "$MODE" in
        lab) protected+=(state research knowledge) ;;
        review) protected=(agent .github requirements.txt self knowledge research experiments) ;;
    esac
    git checkout -q HEAD -- "${protected[@]}"
    git clean -fdq -- "${protected[@]}"

    # Leave out files too big for the repository.
    git ls-files -z -m -o --exclude-standard | while IFS= read -r -d '' f; do
        if [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 5242880 ]; then
            echo "Not saved (over 5 MB): $f"
            git checkout -q -- "$f" 2>/dev/null || rm -f "$f"
        fi
    done

    git add -A
    local label="Research session"
    [ "$MODE" = lab ] && label="Lab run"
    git -c core.hooksPath=/dev/null -c user.name=agi-research-agent \
        -c user.email=agi-research-agent@users.noreply.github.com \
        commit -qm "$label $(date -u +'%Y-%m-%d %H:%M')" || return 0
    git -c core.hooksPath=/dev/null pull -q --rebase "$AGIEXP_REMOTE" "$AGIEXP_BRANCH"
    git push -q "$AGIEXP_REMOTE" "HEAD:$AGIEXP_BRANCH"
}

main "$@"
exit
