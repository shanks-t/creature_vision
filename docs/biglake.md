### Big Query notes for managing data
- set up big lake connection:
```
bq mk --connection --location=US --project_id=creature-vision \
    --connection_type=CLOUD_RESOURCE creature-gcs-connection
```

