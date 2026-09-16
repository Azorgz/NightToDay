#!/bin/bash
PROJECT_NAME="pr-remote-sensing-1a"

REMOTE_PATH="/home/godeta/PycharmProjects/TIR2VIS/datasets/"
LOCAL_PATH="cargo.ciment:/silenus/PROJECTS/${PROJECT_NAME}/godeta/datasets/"

rsync -avxH "$REMOTE_PATH" "$LOCAL_PATH"