#!/usr/bin/env bash
set -u

LOCK_FILE="/mnt/disks/data/ingest.lock"
STATE_FILE="/tmp/idle_since.txt"
IDLE_THRESHOLD_SECONDS=2700  # 45 minutes of continuous idle
CPU_THRESHOLD=8              # CPU % below 8% is considered idle

# 1. Skip if active interactive SSH user session exists
if who | grep -q 'pts/'; then
    rm -f "$STATE_FILE"
    exit 0
fi

# 2. Skip if active ingestion lock is held by a running process
if [ -f "$LOCK_FILE" ] && ! flock -n "$LOCK_FILE" true 2>/dev/null; then
    rm -f "$STATE_FILE"
    exit 0
fi

# 3. Check CPU usage over a 2-second top sample
CPU_IDLE=$(top -bn2 -d 1 | grep "Cpu(s)" | tail -n 1 | awk -F',' '{for(i=1;i<=NF;i++) if($i ~ /id/) print $i}' | awk '{print $1}' | cut -d'.' -f1)
CPU_IDLE=${CPU_IDLE:-100}
CPU_USAGE=$((100 - CPU_IDLE))

if [ "$CPU_USAGE" -ge "$CPU_THRESHOLD" ]; then
    # Active workload detected; reset timer
    rm -f "$STATE_FILE"
    exit 0
fi

# 4. Machine is idle: record or advance idle timer
NOW=$(date +%s)
if [ ! -f "$STATE_FILE" ]; then
    echo "$NOW" > "$STATE_FILE"
    exit 0
fi

IDLE_START=$(cat "$STATE_FILE" 2>/dev/null || echo "")
if ! [[ "$IDLE_START" =~ ^[0-9]+$ ]]; then
    echo "$NOW" > "$STATE_FILE"
    exit 0
fi

ELAPSED=$((NOW - IDLE_START))

if [ "$ELAPSED" -ge "$IDLE_THRESHOLD_SECONDS" ]; then
    echo "$(date -Is) - Idle for ${ELAPSED}s (>= ${IDLE_THRESHOLD_SECONDS}s). Triggering safe auto-shutdown." >> /var/log/auto_shutdown.log
    rm -f "$STATE_FILE"
    /sbin/shutdown -h now "Auto-shutdown: system idle for 45 minutes"
fi
