#!/bin/bash
# Removes pred.npy and pred_prob.npy from every scene subfolder in a split directory.
# Usage: bash cleanup_preds.sh <split_dir>
# Example: bash cleanup_preds.sh /workspace/data/combined/val

SPLIT_DIR="$1"

if [ -z "$SPLIT_DIR" ]; then
    echo "Usage: bash cleanup_preds.sh <split_dir>"
    exit 1
fi

if [ ! -d "$SPLIT_DIR" ]; then
    echo "Error: directory not found: $SPLIT_DIR"
    exit 1
fi

COUNT=0
for SCENE_DIR in "$SPLIT_DIR"/*/; do
    for FILE in pred.npy pred_prob.npy; do
        TARGET="$SCENE_DIR$FILE"
        if [ -f "$TARGET" ]; then
            rm "$TARGET"
            echo "Deleted: $TARGET"
            COUNT=$((COUNT + 1))
        fi
    done
done

echo "Done. $COUNT file(s) removed."