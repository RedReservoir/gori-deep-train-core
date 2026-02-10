class BaseEarlyStopper:
    """
    Base class for early stopper objects.

    An early stopper defines a policy for stopping the training pipeline, and for keeping track
    of the best model epoch so far.
    """


    def update(
        self,
        loss_reg_pool,
        loss_weighter
    ):
        """
        Updates the internal state of the early stopper with the last epoch data.
        This method is meant to be called once per epoch, after the validation loop.
        The first time this method is called (during epoch 0, where only validation is performed),
            it must be done for initialization purposes.

        :param loss_reg_pool: dict of str -> gorideep.utils.loss_register.LossRegister
            Loss register pool of the training pipeline.
        :param loss_weighter: gorideep.loss_weighters.LossWeighter
            Loss weighter of the training pipeline.
        """

        raise NotImplementedError


    def early_stop(
        self
    ):
        """
        Determines whether the training should be stopped.
        Calling this method must not modify internal state data.

        :return: bool
            True iff the training must be stopped.
        """

        raise NotImplementedError


    def improvement(
        self
    ):
        """
        Determines whether the model improved during the last epoch using.
        Calling this method must not modify internal state data.

        :return: bool
            True iff the last epoch is considered an improvement.
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
