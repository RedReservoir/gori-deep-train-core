class BaseDataTransform:
    """
    Base data transform class for performing raw data transformations in datasets.
    Subclasses of this class are expected to implement the `__call__` method.

    :param logger: any, optional
        Logger object in case logging are needed.
    """


    def __init__(
        self,
        logger=None
    ):

        self._logger = logger


    def __call__(
        self,
        data_point
    ):
        """
        :param data_batch: dict of str -> any
            The data point to process.
            Its entries may be read, modified, added or removed.
        """

        raise NotImplementedError()
