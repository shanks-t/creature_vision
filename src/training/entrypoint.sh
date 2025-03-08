#!/bin/bash
# Create health check file
touch /tmp/healthy

# Trap SIGTERM and forward it to the Python process
trap 'kill -TERM $PID' TERM INT

# Start the Python process
python -m main &
PID=$!

# Wait for the Python process to complete
wait $PID
