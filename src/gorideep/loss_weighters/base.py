import goripy.file.json



class BaseLossWeighter:
    """
    Base class for loss weighter objects.

    A loss weighter is tasked with managing and updating loss weights.

    The current implementation allows for loss weight updates:
        - During both training and validation loops.
        - At step and / or epoch level.
        - Based only on loss values (not gradients or metrics).
    """


    def event_before_train_epoch(
        self,
        loss_reg_pool
    ):
        """
        Called before the train loop of each epoch.

        :param loss_reg_pool: dict
            Dict with all loss registers.
        """

        pass


    def event_after_train_step(
        self,
        loss_reg_pool
    ):
        """
        Called after every step of the train loop.

        :param loss_reg_pool: dict
            Dict with all loss registers.
        """

        pass


    def event_after_train_epoch(
        self,
        loss_reg_pool
    ):
        """
        Called after the train loop of each epoch.

        :param loss_reg_pool: dict
            Dict with all loss registers.
        """

        pass


    def event_before_val_epoch(
        self,
        loss_reg_pool
    ):
        """
        Called before the validation loop of each epoch.

        :param loss_reg_pool: dict
            Dict with all loss registers.
        """

        pass


    def event_after_val_epoch(
        self,
        loss_reg_pool
    ):
        """
        Called after the validation loop of each epoch.

        :param loss_reg_pool: dict
            Dict with all loss registers.
        """

        pass


    def synchronize(
        self
    ):
        """
        Broadcasts internal state data from rank 0 to all other ranks.
        """

        raise NotImplementedError
    

    def get_loss_weight(
        self,
        loss_reg_key
    ):
        """
        Retrieves the weight for a specific loss.
        Calling this method must not modify internal state data.

        :param loss_reg_key: str
            Key of the loss to retrieve the weight for.
        """

        raise NotImplementedError


    def save_loss_weights(
        self,
        filename,
        loss_reg_key_list
    ):
        """
        Saves all current epoch loss weight data.

        :param dirname: str
            Name of the directory to save into.
            The directory must exist or this method will fail.
        :param loss_reg_key_list: list of str
            Keys of the loss to save the weights of.
        """

        goripy.file.json.save_json(
            {
                loss_reg_key: self.get_loss_weight(loss_reg_key)
                for loss_reg_key in loss_reg_key_list
            },
            filename
        )
    

    def save(
        self,
        dirname
    ):
        """
        Saves internal state data into a directory.

        :param dirname: str
            Name of the directory to save into.
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
            Name of the directory to load from.
            The directory must exist or this method will fail.
        """

        raise NotImplementedError
