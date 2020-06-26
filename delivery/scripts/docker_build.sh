#!/bin/bash

SCRIPTFOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOTFOLDER="$SCRIPTFOLDER/../.."

# shellcheck source=./docker_config.sh
. "$SCRIPTFOLDER/docker_config.sh"

docker build -t "$TAG" "$ROOTFOLDER"
