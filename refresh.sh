#!/bin/bash
# ============================================================================
# Weekly refresh with safety net.
#
# What this does, in order:
#   1. Checks free disk space. Aborts (no changes made) if too low.
#   2. Asks Hugging Face for the latest commit hash of each repo.
#   3. If nothing changed since last successful refresh, exits early —
#      no restart, no risk.
#   4. If something changed, restarts both services PINNED to those new
#      commit hashes (not just "latest"), so we know exactly what we tested.
#   5. Health-checks both services after restart.
#   6. If healthy: remembers these hashes as "last known good".
#      If unhealthy: automatically rolls back to the last known-good
#      hashes and restarts again, so the live site self-heals instead of
#      staying broken.
#
# Optional: set NOTIFY_TOPIC in refresh.env to get a free push notification
# (via ntfy.sh, no signup needed) whenever a refresh fails or rolls back.
# ============================================================================

set -u

DEPLOY_DIR="$HOME/deploy/DRUG-AWS"
LOG_FILE="$HOME/refresh.log"
STATE_FILE="$DEPLOY_DIR/.last-good-revisions"
CONFIG_FILE="$DEPLOY_DIR/refresh.env"

MIN_FREE_MB=2000   # abort if less than ~2GB free
HEALTH_WAIT_SECONDS=40

MODEL_REPO="yogeshagowda/mtech-model"
DATASET_REPO="yogeshagowda/drug-reviews-final-dataset"
EMBEDDINGS_REPO="yogeshagowda/drug-reviews-embeddings"

# Load optional notification config if present (NOTIFY_TOPIC=...)
if [ -f "$CONFIG_FILE" ]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" >> "$LOG_FILE"
}

notify() {
    # Best-effort push notification via ntfy.sh. Silently does nothing if
    # NOTIFY_TOPIC isn't set. To enable: pick any unguessable topic name,
    # put NOTIFY_TOPIC=your-topic-name in refresh.env, and subscribe to
    # that topic in the ntfy app (Android/iOS) or at https://ntfy.sh/your-topic-name
    if [ -n "${NOTIFY_TOPIC:-}" ]; then
        curl -sf -d "$1" "https://ntfy.sh/$NOTIFY_TOPIC" > /dev/null 2>&1 || true
    fi
}

get_latest_sha() {
    # $1 = repo id, $2 = "model" or "dataset"
    local repo="$1"
    local repo_type="$2"
    local url
    if [ "$repo_type" = "dataset" ]; then
        url="https://huggingface.co/api/datasets/$repo"
    else
        url="https://huggingface.co/api/models/$repo"
    fi
    curl -sf "$url" | python3 -c "import sys, json; print(json.load(sys.stdin).get('sha', ''))" 2>/dev/null
}

health_check() {
    curl -sf http://localhost:8001/health > /dev/null 2>&1 && \
    curl -sf http://localhost:8002/health > /dev/null 2>&1
}

restart_with_revisions() {
    # $1=model_rev $2=dataset_rev $3=embeddings_rev
    export MODEL_REVISION="$1"
    export DATASET_REVISION="$2"
    export EMBEDDINGS_REVISION="$3"
    cd "$DEPLOY_DIR" || return 1
    docker compose up -d >> "$LOG_FILE" 2>&1
}

# ---------------------------------------------------------------------------
log "----- Refresh started -----"

# 1. Disk space guard
AVAIL_MB=$(df / --output=avail -m | tail -1 | tr -d ' ')
log "Free disk space: ${AVAIL_MB}MB"
if [ "$AVAIL_MB" -lt "$MIN_FREE_MB" ]; then
    log "ABORT: only ${AVAIL_MB}MB free (need at least ${MIN_FREE_MB}MB). Skipping refresh entirely to avoid making things worse."
    notify "Drug API refresh ABORTED: low disk space (${AVAIL_MB}MB free)"
    exit 1
fi

# 2. Check latest revisions on Hugging Face
NEW_MODEL_REV=$(get_latest_sha "$MODEL_REPO" "model")
NEW_DATASET_REV=$(get_latest_sha "$DATASET_REPO" "dataset")
NEW_EMBEDDINGS_REV=$(get_latest_sha "$EMBEDDINGS_REPO" "dataset")

if [ -z "$NEW_MODEL_REV" ] || [ -z "$NEW_DATASET_REV" ] || [ -z "$NEW_EMBEDDINGS_REV" ]; then
    log "ABORT: could not reach Hugging Face Hub to check for updates (network issue or HF outage). Leaving current deployment untouched."
    notify "Drug API refresh ABORTED: could not reach Hugging Face"
    exit 1
fi

log "Latest revisions - model: $NEW_MODEL_REV | dataset: $NEW_DATASET_REV | embeddings: $NEW_EMBEDDINGS_REV"

# 3. Compare against last known-good revisions
if [ -f "$STATE_FILE" ]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
else
    GOOD_MODEL_REV=""
    GOOD_DATASET_REV=""
    GOOD_EMBEDDINGS_REV=""
fi

if [ "$NEW_MODEL_REV" = "${GOOD_MODEL_REV:-}" ] && \
   [ "$NEW_DATASET_REV" = "${GOOD_DATASET_REV:-}" ] && \
   [ "$NEW_EMBEDDINGS_REV" = "${GOOD_EMBEDDINGS_REV:-}" ]; then
    log "No changes since last successful refresh. Nothing to do."
    exit 0
fi

log "Change detected. Deploying new revisions..."

# 4. Restart pinned to the new revisions
if ! restart_with_revisions "$NEW_MODEL_REV" "$NEW_DATASET_REV" "$NEW_EMBEDDINGS_REV"; then
    log "ERROR: docker compose up failed outright."
    notify "Drug API refresh FAILED: docker compose error, see refresh.log"
    exit 1
fi

# 5. Give both services time to actually load the model/dataset, then check
log "Waiting ${HEALTH_WAIT_SECONDS}s for services to finish loading..."
sleep "$HEALTH_WAIT_SECONDS"

CLASSIFIER_RESTARTS=$(docker inspect classifier --format='{{.RestartCount}}' 2>/dev/null || echo "unknown")
CHATBOT_RESTARTS=$(docker inspect chatbot --format='{{.RestartCount}}' 2>/dev/null || echo "unknown")
log "Restart counts after deploy - classifier: $CLASSIFIER_RESTARTS | chatbot: $CHATBOT_RESTARTS"

if health_check; then
    log "SUCCESS: both services healthy on new revisions. Saving as last known-good."
    {
        echo "GOOD_MODEL_REV=$NEW_MODEL_REV"
        echo "GOOD_DATASET_REV=$NEW_DATASET_REV"
        echo "GOOD_EMBEDDINGS_REV=$NEW_EMBEDDINGS_REV"
    } > "$STATE_FILE"
    notify "Drug API refreshed successfully to new model/dataset version"
else
    log "WARNING: health check FAILED after deploying new revisions."

    if [ -n "${GOOD_MODEL_REV:-}" ]; then
        log "Rolling back to last known-good revisions - model: $GOOD_MODEL_REV | dataset: $GOOD_DATASET_REV | embeddings: $GOOD_EMBEDDINGS_REV"
        restart_with_revisions "$GOOD_MODEL_REV" "$GOOD_DATASET_REV" "$GOOD_EMBEDDINGS_REV"
        sleep "$HEALTH_WAIT_SECONDS"
        if health_check; then
            log "Rollback successful. Service restored to last known-good version."
            notify "Drug API: new version failed, automatically ROLLED BACK to last working version"
        else
            log "CRITICAL: rollback ALSO failed health check. Manual intervention needed."
            notify "CRITICAL: Drug API is DOWN and rollback also failed. Check the server now."
        fi
    else
        log "CRITICAL: no prior known-good revision exists to roll back to (this may be the first-ever refresh). Manual intervention needed."
        notify "CRITICAL: Drug API health check failed on first refresh, no rollback available. Check the server now."
    fi
fi

log "----- Refresh finished -----"