#!/bin/bash
PROJECT_NAME="pr-miai-phaims"

REMOTE_PATH="/home/godeta/PycharmProjects/TIR2VIS/datasets/"
LOCAL_PATH="godeta@cargo.univ-grenoble-alpes.fr:/silenus/PROJECTS/${PROJECT_NAME}/godeta/datasets/"

rsync -avxH "$REMOTE_PATH" "$LOCAL_PATH"


