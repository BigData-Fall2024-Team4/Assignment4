#!/bin/bash
set -e

# Start chrony in the background if we're running as root
if [ "$(id -u)" = "0" ]; then
    chronyd -d &
    sleep 2
fi

# Find and execute the original Airflow entrypoint
AIRFLOW_ENTRYPOINT="/usr/bin/dumb-init -- /entrypoint"
if [ -f "$AIRFLOW_ENTRYPOINT" ]; then
    exec $AIRFLOW_ENTRYPOINT "$@"
else
    echo "Error: Could not find Airflow entrypoint script"
    exit 1
fi