#!/usr/bin/env bash
# Local Weights & Biases server (self-hosted) via Docker Compose.
#
#   ./manage.sh up            start the server   -> http://wandb-local.localhost:8421
#   ./manage.sh down          stop the server
#   ./manage.sh status        state + autostart + URL
#   ./manage.sh logs          follow server logs
#   ./manage.sh enable-boot   autostart on boot (systemd system unit, sudo once)
#   ./manage.sh disable-boot  remove autostart
#
# License (free, 1 user) is read from ~/.keys/wandb_local_license and injected as
# the container's LICENSE env var, so the system-admin panel isn't needed at all.
#
# First run: open http://wandb-local.localhost:8421, create the local account
# (first signup becomes admin), copy the API key, then point the client at it:
#   pip install wandb
#   wandb login --host=http://wandb-local.localhost:8421 <API_KEY>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE=wandb-local
UNIT=/etc/systemd/system/${SERVICE}.service
URL=http://wandb-local.localhost:8421
LICENSE_FILE="${WANDB_LICENSE_FILE:-$HOME/.keys/wandb_local_license}"
cd "$SCRIPT_DIR"

# Once boot is enabled, route start/stop through systemd so there's one owner;
# otherwise talk to compose directly.
service_installed() { [ -f "$UNIT" ]; }

# Load the license token (kept outside the repo) into WANDB_LICENSE; compose maps
# it to the container's LICENSE var. Whitespace/newline stripped.
load_license() {
  if [ -s "$LICENSE_FILE" ]; then
    export WANDB_LICENSE="$(tr -d '[:space:]' < "$LICENSE_FILE")"
  else
    echo "warn: $LICENSE_FILE missing — running unlicensed (4-user cap)." >&2
  fi
}

# __up/__down are the real compose actions (used directly by the systemd unit so
# the license is loaded at boot too). up/down are the user-facing wrappers.
__up()   { load_license; docker compose up -d; }
__down() { docker compose down; }

# Print the leading comment block (everything above `set -euo pipefail`) as help.
usage() { sed -n '2,/^set /{/^set /d;s/^# \{0,1\}//;s/^#$//;p}' "$0"; }

case "${1:-}" in
  help|-h|--help)
    usage
    ;;
  up)
    if service_installed; then sudo systemctl start "$SERVICE"; else __up; fi
    echo "W&B local starting at ${URL} (first launch can take ~1 min to migrate)."
    echo "Then: wandb login --host=${URL} <API_KEY>"
    ;;
  down)
    if service_installed; then sudo systemctl stop "$SERVICE"; else __down; fi
    ;;
  __up)   __up ;;    # internal: invoked by the systemd unit
  __down) __down ;;  # internal: invoked by the systemd unit
  status)
    docker compose ps
    [ -s "$LICENSE_FILE" ] && echo "lic:  present" || echo "lic:  MISSING ($LICENSE_FILE)"
    if service_installed; then
      echo "boot: $(systemctl is-enabled "$SERVICE" 2>/dev/null || echo unknown)"
    else
      echo "boot: not installed"
    fi
    echo "URL:  ${URL}"
    ;;
  logs)
    docker compose logs -f
    ;;
  enable-boot)
    sudo tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=Weights & Biases local server
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${SCRIPT_DIR}
# Absolute license path baked in — don't rely on systemd providing \$HOME.
# The client keys (WANDB_BASE_URL/WANDB_API_KEY) are NOT needed here; the server
# doesn't read them, they're for the training process.
Environment=WANDB_LICENSE_FILE=${HOME}/.keys/wandb_local_license
ExecStart=${SCRIPT_DIR}/manage.sh __up
ExecStop=${SCRIPT_DIR}/manage.sh __down
User=${USER}
Group=docker

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable --now "$SERVICE"
    echo "Autostart enabled — server will start with the computer (${URL})."
    ;;
  disable-boot)
    sudo systemctl disable --now "$SERVICE" 2>/dev/null || true
    sudo rm -f "$UNIT"
    sudo systemctl daemon-reload
    echo "Autostart removed. Use './manage.sh up' to run manually."
    ;;
  *)
    [ -n "${1:-}" ] && echo "unknown command: $1" >&2
    usage >&2
    exit 1
    ;;
esac

