#!/bin/bash

# ========== CONFIGURATION ==========

MAX_AGENTS=5
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/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"  # <-- 본인 Webhook으로 수정

# cal-07 에서는 Image load 데이터 실행 금지: ImageNet-R, RESISC45, CropDisease X
SWEEP_KEYS_ORDERED=(
    "mmea-owcl/Experimental Results on the MMEA-OWCL/v5sx5t6b"  # iCaRL
    "mmea-owcl/Experimental Results on the MMEA-OWCL/fw5m3pnu"  # EWC
    "mmea-owcl/Experimental Results on the MMEA-OWCL/cib4aa48"  # LwF
)


# Sweep 실행 횟수를 지정하는 연관 배열
declare -A SWEEP_REPEAT_COUNTS=(
    ["mmea-owcl/Experimental Results on the MMEA-OWCL/v5sx5t6b"]=2  # iCaRL
    ["mmea-owcl/Experimental Results on the MMEA-OWCL/fw5m3pnu"]=2  # EWC
    ["mmea-owcl/Experimental Results on the MMEA-OWCL/cib4aa48"]=2  # LwF
)


# ========== FUNCTIONS ==========


count_agents() {
    pgrep -af "wandb agent" | wc -l
}


launch_agent() {
    local SWEEP_ID=$1
    local OPTIONS=$2
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching agent: $SWEEP_ID $OPTIONS"
    nohup wandb agent "$SWEEP_ID" $OPTIONS &
    sleep 10
}


wait_for_slot() {
    while [ "$(count_agents)" -ge "$MAX_AGENTS" ]; do
        echo "Waiting for available slot..."
        sleep 60
    done
}


send_slack_notification() {
    local SWEEP_ID="$1"
    curl -X POST -H 'Content-type: application/json' \
         --data "{\"text\":\"✅ Sweep completed: \`$SWEEP_ID\` on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S')\"}" \
         "$SLACK_WEBHOOK_URL"
}


monitor_sweep() {
    local SWEEP_ID="$1"
    while true; do
        local AGENT_COUNT
        AGENT_COUNT=$(pgrep -af "wandb agent $SWEEP_ID" | wc -l)
        if [ "$AGENT_COUNT" -eq 0 ]; then
            send_slack_notification "$SWEEP_ID"
            break
        fi
        sleep 30
    done
}


# ========== MAIN ==========

echo "========== Run sweeps with per-sweep repeat count =========="

for SWEEP_ID in "${SWEEP_KEYS_ORDERED[@]}"; do
    COUNT=${SWEEP_REPEAT_COUNTS[$SWEEP_ID]}
    for ((i=1; i<=COUNT; i++)); do
        wait_for_slot
        echo "Launching $SWEEP_ID (run $i/$COUNT)"
        launch_agent "$SWEEP_ID"
    done
    monitor_sweep "$SWEEP_ID" & # 알림 병렬 감시
done

wait
echo "✅ All sweeps completed!"