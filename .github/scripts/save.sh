#!/usr/bin/env bash
# Save a workflow's results to the repository: commit only the paths given, then push,
# retrying a few times if another workflow pushed first. If it still can't push, it pushes
# to a rescue branch so nothing is lost, and fails the run so it shows red.
#
# Usage: .github/scripts/save.sh "<commit message>" <path>...
# Needs GITHUB_TOKEN, GITHUB_REPOSITORY and GITHUB_REF_NAME in the environment.
set -uo pipefail
msg="$1"; shift

# Never save symlinks from the agent's folders: trusted code could later follow one to a
# secret (such as /proc/self/environ) or to git's settings.
find self knowledge research experiments -type l -print -delete 2>/dev/null || true

# Drop anything too big for the repository.
git ls-files -z -m -o --exclude-standard -- "$@" | while IFS= read -r -d '' f; do
  if [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 5242880 ]; then
    echo "Not saved (over 5 MB): $f"
    git checkout -q -- "$f" 2>/dev/null || rm -f "$f"
  fi
done

git config user.name "agi-research-agent"
git config user.email "agi-research-agent@users.noreply.github.com"
for p in "$@"; do
  if [ -e "$p" ] || git ls-files --error-unmatch -- "$p" > /dev/null 2>&1; then git add -A -- "$p"; fi
done
if git diff --cached --quiet; then
  echo "Nothing to save."
  exit 0
fi
git -c core.hooksPath=/dev/null commit -qm "$msg" || { echo "::error::git commit failed"; exit 1; }

# The token goes to git through a credential helper that reads it from the environment, so
# it never appears on a command line, where other processes on the machine could see it.
export GIT_TOKEN="$GITHUB_TOKEN"
git config credential.helper '!f() { echo username=x-access-token; echo "password=$GIT_TOKEN"; }; f'
REMOTE="${SAVE_REMOTE:-https://github.com/${GITHUB_REPOSITORY}.git}"  # SAVE_REMOTE: for offline tests
GIT="git -c core.hooksPath=/dev/null"

for attempt in 1 2 3 4 5; do
  # --autostash: files changed but not being saved would otherwise stop the rebase.
  if $GIT pull -q --rebase --autostash "$REMOTE" "$GITHUB_REF_NAME" && $GIT push -q "$REMOTE" "HEAD:$GITHUB_REF_NAME"; then
    echo "Saved to $GITHUB_REF_NAME."
    exit 0
  fi
  $GIT rebase --abort 2>/dev/null || true
  echo "Save attempt $attempt failed; retrying."
  sleep $((attempt * ${SAVE_RETRY_SECONDS:-15}))
done

branch="rescue/$(echo "$GITHUB_WORKFLOW" | tr -c 'A-Za-z0-9\n' '-')-${GITHUB_RUN_ID}"
if $GIT push -q "$REMOTE" "HEAD:refs/heads/$branch"; then
  echo "::error::Could not save to $GITHUB_REF_NAME (probably a conflict with another workflow). Nothing is lost: the results are on the branch $branch. Merge it by hand."
else
  echo "::error::Could not save the results anywhere."
fi
exit 1
