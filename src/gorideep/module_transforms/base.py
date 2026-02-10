class BaseModuleTransform:
    """
    Base module transform class for evaluating modules with tensors and computing losses.
    Subclasses of this class are expected to implement the `__call__` method.

    :param data_counter_pool: dict of str -> gorideep.data_counters.base.BaseDataCounter
        The pool of data counters filled with the datasets.
        Must be treated as read-only.
    :param device: torch.device
        PyTorch device to send tensors to.
    :param logger: any, optional
        Logger object in case logging are needed.
    """


    def __init__(
        self,
        data_counter_pool,
        device,
        logger=None
    ):
        
        self._device = device
        self._logger = logger
        

    def __call__(
        self,
        data_batch,
        module_pool
    ):
        """
        :param data_batch: dict of str -> any
            The data batch to process.
            Its entries may be read, modified, added or removed.
        :param module_pool: dict of str -> torch.nn.Module
            The pool of modules to train.
            Must be treated as read-only.
        """

        raise NotImplementedError()
