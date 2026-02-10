class BaseDataCounter:
    """
    Base class for data counters.
    """


    def count(
        self,
        metadata_point
    ):
        """
        Accumulates data counts from one dataset item.

        :param metadata_point: dict
            A dict containing metadata information of a dataset item.
        """

        raise NotImplementedError


    def save(
        self,
        dirname
    ):
        """
        Saves internal state data into a directory.

        :param dirname: str
            Name of the directory to save internal state data into.
            The directory must exist or this method will fail.
        """

        raise NotImplementedError


    def load(
        self,
        dirname
    ):
        """
        Loads internal state data from a directory.

        :param dirname: str
            Name of the directory to load internal state data from.
            The directory must exist or this method will fail.
        """

        raise NotImplementedError
