#!/bin/bash

REMOTE_PATH="godeta@cargo.univ-grenoble-alpes.fr:/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/FoalGAN_FLIR/web/"
LOCAL_PATH="$HOME/Images/result-bigfoot/web"

rsync -avxH -c "$REMOTE_PATH" "$LOCAL_PATH"
