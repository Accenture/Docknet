class IncompatibleInputDimensionException(Exception):
    def __init__(self, actual_dimension, expected_dimension):
        super().__init__("Expected input arrays of size {} but got arrays of size {}".format(expected_dimension,
                                                                                             actual_dimension))


class IncompatibleSizeArrayException(Exception):
    def __init__(self, actual_size, expected_size):
        super().__init__("Expected array of size {} but got array of size {}".format(expected_size, actual_size))
