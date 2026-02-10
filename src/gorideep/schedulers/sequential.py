import os
import shutil

import numpy

import goripy.file.json

from gorideep.schedulers.base import BaseLRScheduler



class SequentialLRScheduler(BaseLRScheduler):
    """
    Applies multiple LR schedulers sequentially.

    :param optimizer: torch.optim.Optimizer
        Optimizer to manage with this scheduler.
    
    :param schedulers: list of BaseLRScheduler
        List of LR schedulers to apply.

    :param start_epochs: list of int
        List of start epoch values to initialize each LR scheduler on.
    
    :param milestones: list of int
        Target epoch number to reach until switching to the next scheduler.
        Must contain as many elements as schedulers minus one.
        The last scheduler will be applied indefinitely.

    """

    def __init__(
        self,
        optimizer,
        schedulers,
        start_epochs,
        milestones
    ):

        super().__init__(optimizer)

        self._optimizer = optimizer
        self._schedulers = schedulers
        self._start_epochs = start_epochs
        self._milestones = milestones

        # Initialize internal state

        self._curr_epoch = 1

        self._curr_sched_idx = 0
        self._curr_sched_base_lr_dict = None

        self._init_next_sched = True


    def initialize(
        self,
        start_epoch=1
    ):

        # Apply start_epoch

        if start_epoch > self._curr_epoch:

            self._curr_epoch = start_epoch


    ########


    def event_before_train_epoch(
        self,
        epoch_num_steps
    ):

        # Advance sub-scheduler

        while \
            (self._curr_sched_idx < len(self._milestones)) and \
            (self._curr_epoch > self._milestones[self._curr_sched_idx]):

            self._curr_sched_idx += 1
            self._init_next_sched = True

        # Initialize next sub-scheduler

        if self._init_next_sched:

            subsched_start_epoch = self._start_epochs[self._curr_sched_idx]
            
            last_milestone_epoch = 0 if self._curr_sched_idx == 0 else self._milestones[self._curr_sched_idx - 1]
            subsched_start_epoch += self._curr_epoch - last_milestone_epoch - 1

            self._curr_sched_base_lr_dict = self._get_optimizer_lrs()
            self._schedulers[self._curr_sched_idx]._base_lr_dict = self._curr_sched_base_lr_dict

            self._schedulers[self._curr_sched_idx].initialize(subsched_start_epoch)

            self._init_next_sched = False

        # Call sub-scheduler update method

        self._schedulers[self._curr_sched_idx].event_before_train_epoch(epoch_num_steps)


    def event_after_train_step(
        self,
        epoch_step_idx
    ):

        # Call sub-scheduler update method

        self._schedulers[self._curr_sched_idx].event_after_train_step(epoch_step_idx)


    def event_after_train_epoch(
        self
    ):

        # Call sub-scheduler update method

        self._schedulers[self._curr_sched_idx].event_after_train_epoch()

        # Update internal state

        self._curr_epoch += 1


    ########


    def save_epoch_lr_data(
        self,
        dirname
    ):

        goripy.file.json.save_json(
            self._schedulers[self._curr_sched_idx]._init_lr_dict,
            os.path.join(dirname, "init_lr_dict.json")
        )

        numpy.savez(
            os.path.join(dirname, "step_lr_arr_dict.npz"),
            **(self._schedulers[self._curr_sched_idx]._step_lr_arr_dict)
        )

        goripy.file.json.save_json(
            self._schedulers[self._curr_sched_idx]._final_lr_dict,
            os.path.join(dirname, "final_lr_dict.json")
        )


    def save(
        self,
        dirname
    ):
        
        # Save internal state of sub-schedulers

        for sched_idx, sched in enumerate(self._schedulers):
            
            sched_subdirname = os.path.join(dirname, "sched_{:d}".format(sched_idx))
            
            if os.path.exists(sched_subdirname): shutil.rmtree(sched_subdirname)
            os.mkdir(sched_subdirname)

            sched.save(sched_subdirname)

        # Save internal state
        
        internal_state_dict = {
            "curr_epoch": self._curr_epoch,
            "curr_sched_idx": self._curr_sched_idx,
            "init_next_sched": self._init_next_sched,
            "curr_sched_base_lr_dict": self._curr_sched_base_lr_dict
        }

        goripy.file.json.save_json(
            internal_state_dict,
            os.path.join(dirname, "internal_state_dict.json")
        )


    def load(
        self,
        dirname
    ):

        # Load internal state of sub-schedulers

        for sched_idx, sched in enumerate(self._schedulers):
            
            sched_subdirname = os.path.join(dirname, "sched_{:d}".format(sched_idx))

            sched.load(sched_subdirname)

        # Load internal state

        internal_state_dict = goripy.file.json.load_json(
            os.path.join(dirname, "internal_state_dict.json")
        )
        
        self._curr_epoch = internal_state_dict["curr_epoch"]
        self._curr_sched_idx = internal_state_dict["curr_sched_idx"]
        self._init_next_sched = internal_state_dict["init_next_sched"]
        self._curr_sched_base_lr_dict = internal_state_dict["curr_sched_base_lr_dict"]

        # Restore sub-scheduler base LRs

        self._schedulers[self._curr_sched_idx]._base_lr_dict = self._curr_sched_base_lr_dict
