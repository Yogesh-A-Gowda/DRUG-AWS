#!/bin/bash
# Restarts both services so they re-check Hugging Face Hub for updates.
# transformers/datasets only re-download files that actually changed,
# so this is cheap on weeks with no new push.

LOG_FILE="$HOME/refresh.log"

echo "----- $(date) -----" >> "$LOG_FILE"
echo "Restarting classifier container..." >> "$LOG_FILE"
docker restart classifier >> "$LOG_FILE" 2>&1

echo "Restarting chatbot container..." >> "$LOG_FILE"
docker restart chatbot >> "$LOG_FILE" 2>&1

echo "Refresh complete." >> "$LOG_FILE"
