import os

import yaml


class Config:
    """
    Loader of the application configuration in resources file config.yaml; this class is defined in a way that prevents
    the configuration file to be loaded more than once, so further uses of the class will return the same dictionary
    of parameters that was loaded the first time
    """
    class __Config:
        """
        Private class used to actually load the config.yaml file.
        """
        def __init__(self, path=None):
            """
            Loads the config.yaml file as a Python dictionary
            :param path: path to the configuration file to load; if None, it will be assumed to be file config.yaml in
            the package resources folder
            """
            if path is None:
                path = os.path.join(os.path.dirname(__file__), '..', 'resources', 'config.yaml')
            with open(path, 'r', encoding='UTF-8') as fp:
                self.config = yaml.load(fp, Loader=yaml.FullLoader)

        def __str__(self):
            return str(self.config)

    # Class variable shared across all Config instances where the configuration dictionary will be stored
    instance = None

    def __init__(self, path=None):
        """
        Load the config file if it was not already loaded
        :param path: path to the configuration file to load; if None, it will be assumed to be file config.yaml in
        """
        if not Config.instance:
            Config.instance = Config.__Config(path)

    def __getattr__(self, name):
        """
        Returns the specified configuration parameter value
        :param name: the name of the configuration parameter
        :return: the configuration parameter value
        """
        return self.instance.config[name]
