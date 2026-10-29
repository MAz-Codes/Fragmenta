#!/bin/bash
# Live view of a running build_flash_attn.sh: compiled objects out of 73, the
# file being compiled now, time left, and the container's memory. pip runs
# quietly, so progress is read from inside the container, not from the log.
#
# Usage (repo root): packaging/linux/flash_attn_progress.sh [build.log]
LOG="${1:-packaging/build/flash-attn/build.log}"
TOTAL=73          # 72 CUDA kernels + flash_api.cpp
WIDTH=40

fmt() { local s=$1; printf '%dh %02dm' $((s / 3600)) $((s % 3600 / 60)); }
gb() { awk -v b="$1" 'BEGIN { printf "%.1f GB", b / 1073741824 }'; }

while true; do
    C=$(docker ps -q --filter ancestor=quay.io/pypa/manylinux_2_28_x86_64 | head -1)
    [ -z "$ONCE" ] && clear
    echo "flash-attn wheel build"
    echo
    if [ -z "$C" ]; then
        if grep -q '^done:' "$LOG" 2>/dev/null; then
            echo "  finished: $(grep '^done:' "$LOG" | cut -d' ' -f2-)"
            grep -E 'peak container memory' "$LOG"
        elif grep -q 'ERROR' "$LOG" 2>/dev/null; then
            echo "  failed — see $LOG"
            grep -m1 -E 'Killed|error:' "$LOG" | cut -c1-120
        else
            echo "  not running"
        fi
        exit 0
    fi
    IFS='|' read -r n start cur arch anon cap oom now <<< "$(docker exec "$C" sh -c '
        d=$(ls -d /tmp/pip-wheel-*/flash-attn_*/build/temp.linux-x86_64-cpython-3* 2>/dev/null | head -1)
        n=0; start=0
        if [ -n "$d" ]; then
            n=$(find "$d" -name "*.o" | wc -l)
            start=$(stat -c %Y "$d/build.ninja" 2>/dev/null || echo 0)
        fi
        args=$(ps -eo args)
        cur=$(echo "$args" | grep -oE "flash_[a-z0-9_]+\.cu|flash_api\.cpp" | head -1)
        arch=$(echo "$args" | grep -oE "arch compute_[0-9]+" | head -1 | grep -oE "[0-9]+$")
        echo "$n|$start|$cur|$arch|$(grep -E "^anon " /sys/fs/cgroup/memory.stat | cut -d" " -f2)|$(cat /sys/fs/cgroup/memory.max)|$(grep oom_kill /sys/fs/cgroup/memory.events | cut -d" " -f2)|$(date +%s)"')"
    if [ "$start" = "0" ]; then
        echo "  setting up: CUDA toolkit, GCC 13, torch (a few minutes)"
        echo "  $(tail -1 "$LOG" | cut -c1-100)"
    else
        filled=$((n * WIDTH / TOTAL))
        bar=$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' $((WIDTH - filled)) '' | tr ' ' '.')
        printf '  [%s] %d / %d  (%d%%)\n\n' "$bar" "$n" "$TOTAL" $((n * 100 / TOTAL))
        elapsed=$((now - start))
        echo "  compiling   ${cur:-…}${arch:+  (sm_$arch)}"
        echo "  elapsed     $(fmt $elapsed)"
        if [ "$n" -gt 0 ]; then
            echo "  time left   ~$(fmt $(( (TOTAL - n) * elapsed / n )))  (backward kernels come first and are the slowest)"
        fi
    fi
    echo
    echo "  memory      $(gb "$anon") of $(gb "$cap") cap    out-of-memory kills: ${oom:-0}"
    echo
    [ -n "$ONCE" ] && exit 0
    echo "  refreshes every 15 s · Ctrl+C to quit (the build keeps running)"
    sleep 15
done
