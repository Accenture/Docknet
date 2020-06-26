#!/bin/bash

SCRIPTFOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# shellcheck source=./docker_config.sh
. "$SCRIPTFOLDER/docker_config.sh"

docker run -p "$HOST_PORT":"$GUEST_PORT" -it "$TAG"
