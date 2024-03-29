FROM ubuntu:22.04

LABEL docknet.docker.version="1"

# System update
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get dist-upgrade -y

# Set locale
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Python and common tools
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/Dublin apt-get -y install tzdata
RUN apt-get install -y python3.9 python3.9-dev python3.9-venv

# Create Docker user
RUN useradd -ms /bin/bash docker

# Copy the Docknet repo into the Docker container
ADD . /home/docker/docknet
# Make the docker user the docknet folder owner
RUN chown -R docker:docker /home/docker/docknet

# Run build script as the docknet user to install the package and run the tests
USER docker
WORKDIR /home/docker/docknet
RUN delivery/scripts/build.sh

CMD . /home/docker/docknet_venv/bin/activate && docknet_start
