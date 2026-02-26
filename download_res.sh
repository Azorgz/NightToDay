#!/bin/bash

# Get absolute path of the folder containing this script
PROJECT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

#REMOTE_PATH="godeta@cargo.univ-grenoble-alpes.fr:/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/NightToday/"
#LOCAL_PATH="${PROJECT_DIR}/checkpoints/download/"
#echo "Project dir: ${PROJECT_DIR}"
#echo "Syncing to: ${LOCAL_PATH}"
#rsync -avxH -c "$REMOTE_PATH" "$LOCAL_PATH"


REMOTE_PATH="godeta@cargo.univ-grenoble-alpes.fr:/silenus/PROJECTS/pr-remote-sensing-1a/godeta/training_visuals/"
LOCAL_PATH="${PROJECT_DIR}/training_visuals/"

echo "Project dir: ${PROJECT_DIR}"
echo "Syncing to: ${LOCAL_PATH}"

rsync -avxH -c "${REMOTE_PATH}" "${LOCAL_PATH}"
