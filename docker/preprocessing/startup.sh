#!/bin/bash
echo "Starting custom Dataflow container"

# Check if running locally or in Dataflow
if [ "$1" = "local" ]; then
    # Run the pipeline directly for local testing
    python /app/main.py "$@"
else
    # In Dataflow, use the standard boot script
    /opt/apache/beam/boot "$@"
fi
