#!/bin/bash

SCRIPTFOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# shellcheck source=./docker_config.sh
. "$SCRIPTFOLDER/docker_config.sh"

docker rmi -f "$(docker images -f "label=$LABEL" -q)"
