import os

import goripy.file.json

from gorideep.checkpoint_savers.base import BaseCheckpointSaver



class GenericCheckpointSaver(BaseCheckpointSaver):
    """
    Checkpoint saver with generic policy.    

    :param period_active: bool, default=False
        If True, will save checkpoints regularly every few epochs.
        Settings controlled by `period_start` and `period_step`.
    :param period_start: int, default=0
        Required if `period_active` is `True`.
        Number of epochs starting from which to save checkpoints.
    :param period_step: int, default=1
        Required if `period_active` is `True`.
        Number of epochs every which to save checkpoints.
    
    :param improvement_active: bool, default=False
        If True, will save checkpoints every time the early stopper measures a model improvement.
    """


    def __init__(
        self,
        period_active=False,
        period_start=0,
        period_step=1,
        improvement_active=False
    ):
        
        # Arguments

        self._period_active = period_active
        self._period_start = period_start
        self._period_step = period_step
        self._improvement_active = improvement_active

        # Internal state

        self._curr_epoch_num = -1
        self._save_checkpoints = False


    def update(
        self,
        loss_reg_pool,
        loss_weighter,
        early_stopper
    ):

        self._curr_epoch_num += 1
        self._save_checkpoints = False

        if self._period_active:

            if (self._curr_epoch_num - self._period_start) % self._period_step == 0:
                self._save_checkpoints = True
        
        if self._improvement_active:

            if early_stopper.improvement():
                self._save_checkpoints = True


    def save_checkpoints(
        self
    ):
        
        if self._curr_epoch_num == 0: return False
        return self._save_checkpoints


    def save(
        self,
        dirname
    ):
        
        internal_state_dict = {
            "curr_epoch_num": self._curr_epoch_num,
            "save_checkpoints": self._save_checkpoints
        }

        internal_state_filename = os.path.join(dirname, "internal_state.json")
        goripy.file.json.save_json(internal_state_dict, internal_state_filename)


    def load(
        self,
        dirname
    ):

        internal_state_filename = os.path.join(dirname, "internal_state.json")
        internal_state_dict = goripy.file.json.load_json(internal_state_filename)

        self._curr_epoch_num = internal_state_dict["curr_epoch_num"]
        self._save_checkpoints = internal_state_dict["save_checkpoints"]
