#!/bin/bash

SCRIPTFOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# shellcheck source=./docker_config.sh
. "$SCRIPTFOLDER/docker_config.sh"

images="$(docker ps -f "label=$LABEL" -q)"
images=($images)
for image in ${images[@]}
do
  echo "Stopping image $image"
  docker stop $image
done
