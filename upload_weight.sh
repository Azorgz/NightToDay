#!/bin/bash

REMOTE_PATH="$HOME/godeta/Bureau/weight_new/"
#LOCAL_PATH="godeta@cargo.univ-grenoble-alpes.fr:/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/FoalGAN_FLIR/"
LOCAL_PATH="godeta@cargo.univ-grenoble-alpes.fr:/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/CrossRAFT/"

rsync -avxH "$REMOTE_PATH" "$LOCAL_PATH"


