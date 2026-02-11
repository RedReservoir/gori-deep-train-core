class BaseCheckpointSaver:
    """
    Base class for checkpoint saver objects.
    """


    def update(
        self,
        loss_reg_pool,
        loss_weighter,
        early_stopper
    ):
        """
        Updates the internal state of the early stopper with the last epoch data.
        This method is meant to be called once at the end of the validation loop.
        The first time this method is called (during epoch 0, where only validation is performed),
        it must be done for initialization purposes.

        :param loss_reg_pool: dict of str -> gorideep.utils.loss_register.LossRegister
            Loss register pool of the training pipeline.
        :param loss_weighter: gorideep.loss_weighters.LossWeighter
            Loss weighter of the training pipeline.
        :param early_stopper: gorideep.early_stoppers.base.BaseEarlyStopper
            Early stopper of the training pipeline.
        """

        raise NotImplementedError


    def save_checkpoints(
        self
    ):
        """
        Determines whether module checkpoints should be saved.
        Calling this method must not modify internal state data.

        :return: bool
            True iff the module checkpoints must be saved.
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
