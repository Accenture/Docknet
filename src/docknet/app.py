import os

import numpy as np

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from docknet.net import read_pickle
from docknet.util.config import Config

app = Flask(__name__)
api = Api(app)

resources_dir = os.path.join(os.path.dirname(__file__), 'resources')

chessboard_model_pathname = os.path.join(resources_dir, 'chessboard.pkl')
cluster_model_pathname = os.path.join(resources_dir, 'cluster.pkl')
island_model_pathname = os.path.join(resources_dir, 'island.pkl')
swirl_model_pathname = os.path.join(resources_dir, 'swirl.pkl')

config = Config()


class PredictionServer(Resource):
    """
    REST API layer on top of docknet model that returns predicted values, given an input vector defined by 2 parameters
    x0 and x1. This class is to be inherited by another class that provides the path to the Docknet pkl model file to
    use for making predictions.
    """
    def __init__(self, pkl_pathname: str):
        """
        Create a prediction server
        :param pkl_pathname: path to the Docknet pkl model file to load
        """
        super().__init__()
        self.docknet = read_pickle(pkl_pathname)

    def get(self):
        """
        Returns the predicted value for
        :return:
        """
        x0 = request.args.get('x0', type=float)
        x1 = request.args.get('x1', type=float)
        if x0 is None:
            status_code = 400
            success = False
            message = "Missing mandatory argument x0"
        elif x1 is None:
            status_code = 400
            success = False
            message = "Missing mandatory argument x1"
        else:
            success = True
            status_code = 200
            X = np.array([[x0], [x1]])
            Y = np.round(self.docknet.predict(X))
            message = int(Y[0, 0])
        response = jsonify(success=success, message=message)
        response.status_code = status_code
        return response


class ChessboardPredictionServer(PredictionServer):
    def __init__(self):
        super().__init__(chessboard_model_pathname)


class ClusterPredictionServer(PredictionServer):
    def __init__(self):
        super().__init__(cluster_model_pathname)


class IslandPredictionServer(PredictionServer):
    def __init__(self):
        super().__init__(island_model_pathname)


class SwirlPredictionServer(PredictionServer):
    def __init__(self):
        super().__init__(swirl_model_pathname)


# Add the prediction servers for each one of the 4 models chessboard, cluster, island and swirl
api.add_resource(ChessboardPredictionServer, "/chessboard_prediction")
api.add_resource(ClusterPredictionServer, "/cluster_prediction")
api.add_resource(IslandPredictionServer, "/island_prediction")
api.add_resource(SwirlPredictionServer, "/swirl_prediction")


def main():
    # Start the service; the service stops when the process is killed or the docker container running this service is
    # shut down
    app.run(**config.app)


if __name__ == "__main__":
    main()
