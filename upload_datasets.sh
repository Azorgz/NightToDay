#!/bin/bash

REMOTE_PATH="/home/godeta/PycharmProjects/TIR2VIS/datasets/"
LOCAL_PATH="godeta@cargo.univ-grenoble-alpes.fr:/silenus/PROJECTS/pr-remote-sensing-1a/godeta/datasets/"

rsync -avxH "$REMOTE_PATH" "$LOCAL_PATH"


