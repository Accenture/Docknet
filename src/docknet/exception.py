class IncompatibleInputDimensionException(Exception):
    def __init__(self, actual_dimension, expected_dimension):
        super().__init__(f'Expected input arrays of size {expected_dimension} '
                         f'but got arrays of size {actual_dimension}')


class IncompatibleSizeArrayException(Exception):
    def __init__(self, actual_size, expected_size):
        super().__init__(f'Expected array of size {expected_size} but got '
                         f'array of size {actual_size}')
