import os

import goripy.file.json

from gorideep.schedulers.base import BaseLRScheduler



class ExponentialLRScheduler(BaseLRScheduler):
    """
    Implements an exponential LR scheduling policy.

    :param optimizer: torch.nn.optim.Optimizer
        Optimizer to manage with this scheduler.

    :param gamma: float
        Exponential LR decay gamma value.
    """

    def __init__(
        self,
        optimizer,
        gamma
    ):

        super().__init__(optimizer)
        
        self._gamma = gamma

        # Initialize internal state

        self._curr_epoch = 1
        self._first_update = True


    def initialize(
        self,
        start_epoch=1
    ):

        # First update

        if start_epoch > self._curr_epoch:
                
            # LR update

            curr_lr_dict = self._get_optimizer_lrs()

            for param_group_name in curr_lr_dict.keys():
                curr_lr_dict[param_group_name] *= self._gamma ** (start_epoch - self._curr_epoch)

            self._set_optimizer_lrs(curr_lr_dict)

            # Internal state update

            self._curr_epoch = start_epoch


    ########


    def event_before_train_epoch(
        self,
        epoch_num_steps
    ):

        # LR update

        if self._first_update:

            self._first_update = False
        
        else:

            curr_lr_dict = self._get_optimizer_lrs()

            for param_group_name in curr_lr_dict.keys():
                curr_lr_dict[param_group_name] *= self._gamma

            self._set_optimizer_lrs(curr_lr_dict)

        # Call super event

        super().event_before_train_epoch(
            epoch_num_steps
        )


    def event_after_train_step(
        self,
        epoch_step_idx
    ):

        # Call super event

        super().event_after_train_step(
            epoch_step_idx
        )

        
    def event_after_train_epoch(
        self
    ):

        # Internal state update

        self._curr_epoch += 1

        # Call super event

        super().event_after_train_epoch()


    ########


    def save(
        self,
        dirname
    ):
        
        internal_state_dict = {
            "curr_epoch": self._curr_epoch,
            "first_update": self._first_update
        }

        goripy.file.json.save_json(
            internal_state_dict,
            os.path.join(dirname, "internal_state_dict.json")
        )


    def load(
        self,
        dirname
    ):
        
        internal_state_dict = goripy.file.json.load_json(
            os.path.join(dirname, "internal_state_dict.json")
        )
        
        self._curr_epoch = internal_state_dict["curr_epoch"]
        self._first_update = internal_state_dict["first_update"]
