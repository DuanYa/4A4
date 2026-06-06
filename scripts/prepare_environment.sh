#!/usr/bin/env bash
set -u

FRP_DIR="/home/ycf/frp_0.61.0_linux_amd64"
SESSION="frp-0"
CONFIG="frpc.toml"
WATCHDOG_LOG="$FRP_DIR/frp_watchdog.log"

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$WATCHDOG_LOG"
}

tmux_alive() {
  tmux has-session -t "$SESSION" >/dev/null 2>&1
}

frpc_alive() {
  local pid cwd cmd
  for pid in $(pgrep -u "$(id -u)" -x frpc 2>/dev/null); do
    cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if [ "$cwd" = "$FRP_DIR" ] && printf '%s' "$cmd" | grep -q -- "-c $CONFIG"; then
      return 0
    fi
  done
  return 1
}

start_frp() {
  mkdir -p "$FRP_DIR"
  tmux new-session -d -s "$SESSION" -n frpc \
    "cd '$FRP_DIR' && exec ./frpc -c '$CONFIG' >> '$WATCHDOG_LOG' 2>&1"
  log "started frpc in tmux session $SESSION"
}

if tmux_alive && frpc_alive; then
  exit 0
fi

if tmux_alive && ! frpc_alive; then
  log "tmux session $SESSION exists but frpc is not running; recreating session"
  tmux kill-session -t "$SESSION" >/dev/null 2>&1 || true
fi

if ! tmux_alive; then
  log "tmux session $SESSION missing; starting frpc"
  start_frp
fi
