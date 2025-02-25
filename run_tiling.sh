#!/bin/bash

# List of datasets
DATASETS=("cptac_lusc" "cptac_luad")

# Configuration file
CONFIG="histolung/config/datasets_config_ts_224_20x.yaml"

# Number of workers
NUM_WORKERS=24

# Loop over each dataset
for DATASET in "${DATASETS[@]}"; do
    echo "Running preprocessing for dataset: $DATASET"
    
    # Run the preprocessing script
    python histolung/data/preprocess.py tile-wsi-task -c "$CONFIG" --num-workers "$NUM_WORKERS" --dataset "$DATASET"
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Preprocessing failed for $DATASET. Skipping..."
        continue  # Skip to the next dataset
    fi

    echo " Preprocessing completed for $DATASET."
done

echo "All preprocessing tasks completed!"
