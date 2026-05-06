PROJECT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

mkdir -p "${PROJECT_DIR}/checkpoints/download/conf_server"
mkdir -p "${PROJECT_DIR}/checkpoints/download/weights"
mkdir -p "${PROJECT_DIR}/checkpoints/download/visuals"
mkdir -p "${PROJECT_DIR}/checkpoints/download/code"

sshfs godeta@bigfoot.ciment:/home/godeta/NightToDay/NightToday/configs/ "${PROJECT_DIR}/checkpoints/download/conf_server/"
sshfs godeta@bigfoot.ciment:/home/godeta/NightToDay/NightToday/ "${PROJECT_DIR}/checkpoints/download/code/"
sshfs godeta@cargo.univ-grenoble-alpes.fr:/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/NightToday/ "${PROJECT_DIR}/checkpoints/download/weights"
sshfs godeta@cargo.univ-grenoble-alpes.fr:/silenus/PROJECTS/pr-remote-sensing-1a/godeta/training_visuals/ "${PROJECT_DIR}/checkpoints/download/visuals"
#sshfs godeta@cargo.univ-grenoble-alpes.fr:/silenus/PROJECTS/pr-remote-sensing-1a/godeta/datasets/ "${PROJECT_DIR}/checkpoints/download/datasets"