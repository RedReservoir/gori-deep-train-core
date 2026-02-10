import os

import goripy.file.json

from gorideep.schedulers.base import BaseLRScheduler



class LinearStepLRScheduler(BaseLRScheduler):
    """
    Implements a linear LR scheduling policy.

    :param optimizer: torch.nn.optim.Optimizer
        Optimizer to manage with this scheduler.

    :param start_factor: float
        Starting factor for the LR.
    :param end_factor: float
        Ending factor for the LR.

    :param num_epochs: int
        Number of epochs that the linear scheduling should last for.
        Afterwards, the LR will continue with its update trend.
    """

    def __init__(
        self,
        optimizer,
        start_factor,
        end_factor,
        num_epochs,
        start_epoch=1
    ):

        super().__init__(optimizer)
        
        self._start_factor = start_factor
        self._end_factor = end_factor

        self._num_epochs = num_epochs

        # Initialize internal state

        self._curr_epoch = start_epoch


    def initialize(
        self,
        start_epoch=1
    ):
        
        if start_epoch > self._curr_epoch:
                
            self._curr_epoch = start_epoch


    ########


    def event_before_train_epoch(
        self,
        epoch_num_steps
    ):

        # Pre-compute LR ratios

        start_epoch_ratio = (self._curr_epoch - 1.0) / self._num_epochs
        end_epoch_ratio = self._curr_epoch / self._num_epochs

        self._start_epoch_lr_ratio = \
            ((1.0 - start_epoch_ratio) * + self._start_factor) + \
            (start_epoch_ratio * self._end_factor)
        
        self._end_epoch_lr_ratio = \
            ((1.0 - end_epoch_ratio) * + self._start_factor) + \
            (end_epoch_ratio * self._end_factor)

        self._epoch_num_steps = epoch_num_steps
        
        # LR update

        base_lr_dict = self._base_lr_dict.copy()
    
        epoch_step_idx = -1
        curr_step_ratio = (epoch_step_idx + 2.0) / self._epoch_num_steps

        lr_factor = \
            ((1.0 - curr_step_ratio) * + self._start_epoch_lr_ratio) + \
            (curr_step_ratio * self._end_epoch_lr_ratio)
        
        for param_group_name in base_lr_dict.keys():
            base_lr_dict[param_group_name] *= lr_factor

        self._set_optimizer_lrs(base_lr_dict)

        # Call super event
        
        super().event_before_train_epoch(
            epoch_num_steps
        )


    def event_after_train_step(
        self,
        epoch_step_idx
    ):

        # LR update

        if epoch_step_idx < self._epoch_num_steps - 1:

            base_lr_dict = self._base_lr_dict.copy()
        
            curr_step_ratio = (epoch_step_idx + 2.0) / self._epoch_num_steps

            lr_factor = \
                ((1.0 - curr_step_ratio) * + self._start_epoch_lr_ratio) + \
                (curr_step_ratio * self._end_epoch_lr_ratio)
            
            for param_group_name in base_lr_dict.keys():
                base_lr_dict[param_group_name] *= lr_factor

            self._set_optimizer_lrs(base_lr_dict)

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
