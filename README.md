Docknet
=======

The Docknet project comprises:

* a Python package with a pure NumPy implementation of neural networks
* unit tests to validate the code
* a set of Jupyter notebooks making use of the Python package
* a Docker container and REST API to provide an online classification service based on precomputed models

The neural network implementation is strongly based on courses 1 and 2 of Coursera's Deep Learning Specialization: https://www.coursera.org/specializations/deep-learning.

This project has been developed for educational purposes only, namely to:

* understand the math and algorithms required to implement and train neural networks
* illustrate how one could unit test this code
* illustrate how to build a Python package
* illustrate how to consume the Python package with Jupyter notebooks, allowing to mix structured code (the Python package) with exploration code (the notebooks)
* illustrate how to dockerize a Python application and provide a REST API to use it as and online service

Requirements
------------

To run this project Python 3.8 or higher is required, as well as pipenv in order to create a Python virtual environment where to install all the python packages as well as JupyterLab. In Ubuntu, run the command:

`sudo apt-get install python3.8 python3.8-dev python3-venv`

In macOS one can easily install the necessary packages with Homebrew. Homebrew can be installed with the following command:

`/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`

Then one can install the necessary packages as follows:

`brew install python pipenv`

Docker is needed in order to run Docknet as a service inside a Docker container. More information on how to install Docker can be found here:

https://docs.docker.com/get-docker/

Note, the service can also be run without a Docker container, in which case Docker is not needed.

Installation
------------

Run the bash script at:

`delivery/scripts/build.sh`

This script will create a Python virtual environment at:

`$HOME/docknet_venv`

and install there the Docknet Python package along with all the necessary Python dependencies and JupyterLab.

Running the notebooks
---------------------

Activate the Docknet virtual environment:

`source $HOME/docknet_venv/bin/activate`

Go to the main project folder and open JupyterLab with the following command:

`jupyter lab`

A web browser should open with the JupyterLab interface. Navigate to the folder:

`exploration`

4 example notebooks are located there. Each notebook contains a binary classification problem solved with a neural network. The Docknet library contains a set of dataset generators, which produce a random sample for binary classification. A Docknet is created using an appropriate number of layers, neurons and other hyperparameters in order to properly classify the generated data.

Running the web service
-----------------------

There are 2 options for running the web service, running the service directly in your machine or inside a Docker container. For running the service directly in your machine, follow the previous installation steps, then activate the Docknet virtual environment:

`source $HOME/docknet_venv/bin/activate`

then run the command:

`docknet_start`

For running the service inside a Docker container, first go to the project main folder and build the container with the command:

`docker build -t docknet .`

Then run the Docker container with the command:

`docker run -p 8080:8080 -it docknet`

Independently on whether the service is run inside a Docker container or not, 4 classification services will then be available from the following URLs, each one corresponding to one of the classification problems illustrated in the 4 notebooks:

http://localhost:8080/chessboard_prediction?x0=2&x1=2  
http://localhost:8080/cluster_prediction?x0=2&x1=2  
http://localhost:8080/island_prediction?x0=2&x1=2  
http://localhost:8080/swirl_prediction?x0=2&x1=2

Parameters x0 and x1 correspond to the data point to classify where the values (2, 2) have been given as an example. The services return a JSON such as:

{"message": 1, "success": true}

where "message" is either the predicted label (if "success" is true) or the error message (if "success" is false, for instance if there are missing parameters in the URL).

License
-------

Docknet is distributed under the Apache 2.0 license. A copy of the license can be found in the file `LICENSE.txt`.
