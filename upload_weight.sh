#!/bin/bash

REMOTE_PATH="/home/godeta/PycharmProjects/MyTransform/checkpoints/download/latest_net_ResNet"
#REMOTE_PATH="$HOME/Bureau/weight_new/"
#LOCAL_PATH="godeta@cargo.univ-grenoble-alpes.fr:/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/FoalGAN_FLIR/"
LOCAL_PATH="godeta@cargo.univ-grenoble-alpes.fr:/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/NightToday/latest_net_ResNet"

rsync -avxH "$REMOTE_PATH" "$LOCAL_PATH"


