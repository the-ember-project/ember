#!/usr/bin/env bash
# One-time setup for running the agent on your own Linux machine (Ubuntu or Debian)
# instead of GitHub Actions. Run it as your normal user, which needs sudo rights:
#
#   git clone https://github.com/YOUR-NAME/YOUR-REPO.git ~/agiexp
#   bash ~/agiexp/agent/host/install.sh
#
# Safe to run again; it only adds what is missing.

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$HOME/.config/agiexp"
CONFIG="$CONFIG_DIR/env"
CACHE_DIR=/var/cache/agiexp

python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || { echo "Python 3.10 or newer is needed."; exit 1; }
if command -v apt-get >/dev/null; then
    sudo apt-get install -y -qq python3-venv git util-linux procps >/dev/null
fi

# A separate user for the agent's commands, which you may run commands as without a password.
id researcher >/dev/null 2>&1 || sudo useradd -m researcher
echo "$USER ALL=(researcher) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/agiexp >/dev/null
sudo chmod 440 /etc/sudoers.d/agiexp

# Shared cache for downloaded models and packages.
sudo mkdir -p "$CACHE_DIR"
sudo chmod 777 "$CACHE_DIR"

# Settings and secrets, readable only by you.
mkdir -p "$CONFIG_DIR"
if [ ! -f "$CONFIG" ]; then
    cat > "$CONFIG" <<'EOF'
# Settings for the AGI research agent. Keep this file private (chmod 600).
ANTHROPIC_API_KEY=
# A fine-grained GitHub token for the agent's repository only, with read and write access
# to Contents, Issues and Pull requests: github.com/settings/personal-access-tokens
GITHUB_TOKEN=
# The repository as OWNER/NAME
GITHUB_REPOSITORY=
# GitHub usernames allowed to direct the agent, comma-separated: yours, and later your backup person's
AGENT_TRUSTED_USERS=
MONTHLY_BUDGET_USD=40
# AGENT_MODEL=claude-opus-5-5
# The total set aside for the agent (raise it to add funds); it writes a final report and pauses near the end.
TOTAL_FUNDS_USD=100
# Days without contact from you before it writes a final report and pauses (default 180).
# WIND_DOWN_DAYS=180
EOF
fi
chmod 600 "$CONFIG"

# Run a session every day at 06:00, the lab at 08:00, and the review on Sundays at 04:00 (this machine's time zone).
LAUNCH="$REPO_DIR/agent/host/launch.sh"
{
    crontab -l 2>/dev/null | grep -v '# agiexp' || true
    echo "0 6 * * * bash $LAUNCH session >> $CONFIG_DIR/agent.log 2>&1 # agiexp"
    echo "0 8 * * * bash $LAUNCH lab >> $CONFIG_DIR/agent.log 2>&1 # agiexp"
    echo "0 4 * * 0 bash $LAUNCH review >> $CONFIG_DIR/agent.log 2>&1 # agiexp"
} | crontab -

echo "Installed. Next:"
echo "  1. Fill in $CONFIG"
echo "  2. Disable the three workflows in the repository's Actions tab, so the agent doesn't also run there."
echo "  3. Try a session now: bash $LAUNCH session"
echo "Logs go to $CONFIG_DIR/agent.log. To stop: crontab -e and delete the three '# agiexp' lines."
