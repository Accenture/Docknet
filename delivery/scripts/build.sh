#!/bin/bash

PYTHON=`command -v python3.9`
DOCKNET_VENV=$HOME/docknet_venv
SCRIPTFOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOTFOLDER="$SCRIPTFOLDER/../.."
DOCKER_USER="docker"
USER=`whoami`

echo "***************************************"
echo "* Deleting old virtualenv, if present *"
echo "***************************************"

if [ -n "$VIRTUAL_ENV" ]; then
  if [ -n "$(typeset -F | grep -o deactivate)" ]; then
    echo "Deactivating virtualenv"
    deactivate
  else
    echo "Error: cannot deactivate virtualenv; invoke this script with \"source $0\""
    exit 1
  fi
fi
if [ -d "$DOCKNET_VENV" ]; then
   echo "Deleting virtualenv"
   rm -R "${DOCKNET_VENV:?}"
fi

echo "************************************"
echo "* Creating new virtual environment *"
echo "************************************"

$PYTHON -m venv $DOCKNET_VENV

# Activate virtual environment
source "$DOCKNET_VENV"/bin/activate

echo "**********************************"
echo "* Installing global dependencies *"
echo "**********************************"
pip install --upgrade pip==23.0.1
pip install --upgrade setuptools==67.3.3
pip install --upgrade wheel==0.38.4
pip install -r $ROOTFOLDER/requirements.txt
if [ "$USER" != "$DOCKER_USER" ]; then
  pip install -r $ROOTFOLDER/requirements-dev.txt
fi

echo "*****************************"
echo "* Installing Python package *"
echo "*****************************"

cd "$ROOTFOLDER"
pip install .
exit_code=$?
if [ $exit_code != 0 ]; then
	exit $exit_code
fi

echo "*****************************"
echo "* Installed Python packages *"
echo "*****************************"

pip freeze --all

echo "*****************"
echo "* Running tests *"
echo "*****************"

python setup.py test
