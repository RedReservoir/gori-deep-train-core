import os

import numpy
import torch

import goripy.file.json



class BaseLRScheduler:
    """
    Base class for LR scheduler objects.

    The current implementation allows for LR scheduling:

        - During the training loop.
        - At step and / or epoch level.

    :param optimizer: torch.optim.Optimizer
        Optimizer to manage with this scheduler.
    """

    def __init__(
        self,
        optimizer
    ):

        self._optimizer = optimizer
        self._base_lr_dict = self._get_optimizer_lrs()


    def initialize(
        self,
        start_epoch=1
    ):
        """
        Initializes internal state according to a starting epoch.
        Called after the optimizer and scheduler data have been loaded.

        :param start_epoch: int, default=1
            Used to advance the behaviour of this scheduler.
            The first epoch is 1 (default value), meaning standard behaviour.
        """

        raise NotImplementedError


    ########


    def event_before_train_epoch(
        self,
        epoch_num_steps
    ):
        """
        Called before the train loop of each epoch.
        Updates internal state and LRs.
        
        :param epoch_num_steps: int
            Number of expected steps in the following epoch.
        """

        # Register initial LRs

        self._init_lr_dict = self._get_optimizer_lrs()

        # Prepare step LR arrays

        self._step_lr_arr_dict = {
            param_group_name: numpy.empty(shape=(epoch_num_steps), dtype=float)
            for param_group_name in self._init_lr_dict.keys()
        }


    def event_after_train_step(
        self,
        epoch_step_idx
    ):
        """
        Called after each step of the train loop.
        Updates internal state and LRs.

        :param epoch_step_idx: int
            Current step index in the current epoch.
        """

        # Update step LR arrays

        curr_lr_dict = self._get_optimizer_lrs()

        for param_group_name, lr in curr_lr_dict.items():
            self._step_lr_arr_dict[param_group_name][epoch_step_idx] = lr


    def event_after_train_epoch(
        self
    ):
        """
        Called after the train loop of each epoch.
        Updates internal state and LRs.
        """

        # Register final LRs

        self._final_lr_dict = self._get_optimizer_lrs()


    ########


    def _get_optimizer_lrs(
        self
    ):
        """
        Extracts the optimizer parameter group LRs.
        
        :return: dict
            A dict containing all optimizer parameter group current LRs, indexed by
            parameter group name.
        """
        
        lrs_dict = {}

        for param_group in self._optimizer.param_groups:

            param_group_lr = param_group["lr"]
            if isinstance(param_group_lr, torch.Tensor):
                param_group_lr = param_group_lr.item()

            lrs_dict[param_group["name"]] = param_group_lr
        
        return lrs_dict


    def _set_optimizer_lrs(
        self,
        new_lrs_dict
    ):
        """
        Updates the optimizer parameter group LRs.

        :param new_lrs_dict: dict
            A dict containing all optimizer parameter group new LRs, indexed by
            parameter group name.
        """
    
        for param_group in self._optimizer.param_groups:

            new_lr = new_lrs_dict[param_group["name"]]

            if isinstance(param_group["lr"], torch.Tensor):
                param_group["lr"].fill_(new_lr)
            else:
                param_group["lr"] = new_lr


    ########


    def save_epoch_lr_data(
        self,
        dirname
    ):
        """
        Saves internal state data into a directory.

        :param dirname: str
            Name of the directory to save internal state data into.
            The directory must exist or this method will fail.
        """

        goripy.file.json.save_json(
            self._init_lr_dict,
            os.path.join(dirname, "init_lr_dict.json")
        )

        numpy.savez(
            os.path.join(dirname, "step_lr_arr_dict.npz"),
            **(self._step_lr_arr_dict)
        )

        goripy.file.json.save_json(
            self._final_lr_dict,
            os.path.join(dirname, "final_lr_dict.json")
        )


    def save_epoch_lr_data(
        self,
        dirname
    ):
        """
        Saves learning rate data accumulated from the last epoch.

        :param dirname: str
            Name of the directory to save learning rate data into.
            The directory must exist or this method will fail.
        """

        goripy.file.json.save_json(
            self._init_lr_dict,
            os.path.join(dirname, "init_lr_dict.json")
        )

        goripy.file.json.save_json(
            self._final_lr_dict,
            os.path.join(dirname, "final_lr_dict.json")
        )

        numpy.savez(
            os.path.join(dirname, "step_lr_arr_dict.npz")
            **self._step_lr_arr_dict
        )


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

        pass


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

        pass
