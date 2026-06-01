#!/bin/bash

REMOTE_PATH="/home/godeta/PycharmProjects/FusionMethods/fusion_framework/methods/Fusion/NightToDay/checkpoints/latest_net_NightToDay_UResNet"
#REMOTE_PATH="$HOME/Bureau/weight_new/"
#LOCAL_PATH="godeta@cargo.univ-grenoble-alpes.fr:/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/FoalGAN_FLIR/"
LOCAL_PATH="godeta@cargo.univ-grenoble-alpes.fr:/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/NightToday/latest_net_NightToDay_UResNet"
#LOCAL_PATH="godeta@cargo.univ-grenoble-alpes.fr:/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/NightToday/latest_net_NightToDay_UResNet"

rsync -avxH "$REMOTE_PATH" "$LOCAL_PATH"


