#!/bin/bash

#REMOTE_PATH="godeta@cargo.univ-grenoble-alpes.fr:/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/NightToday/"
#LOCAL_PATH="/home/aurelien/PycharmProjects/NightToDay/checkpoints/download/"
#rsync -avxH -c "$REMOTE_PATH" "$LOCAL_PATH"

REMOTE_PATH="godeta@cargo.univ-grenoble-alpes.fr:/silenus/PROJECTS/pr-remote-sensing-1a/godeta/training_visuals/"
LOCAL_PATH="/home/aurelien/PycharmProjects/NightToDay/training_visuals/"
rsync -avxH -c "$REMOTE_PATH" "$LOCAL_PATH"
