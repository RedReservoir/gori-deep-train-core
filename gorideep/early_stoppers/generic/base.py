import os

import goripy.file.json

from gorideep.early_stoppers.base import BaseEarlyStopper



class BaseGenericEarlyStopper(BaseEarlyStopper):
    """
    Early stopper with generic early stopping policy.    

    :param startup: int, default=0
        Number of epochs to wait until early stopping becomes switched on.
    :param patience: int, default=1
        Number of epochs with no improvement until training is stopped.
    :param max_epochs: int, optional
        Maximum number of epochs before the training is stopped.
        If not provided, no limit is imposed on the number of epochs.
    :param abs_tol: float, optional
        Minimum absolute target value change to count as improvement.
        If not provided, 0.0 will be used as default.
        Must not be provided if `rel_tol` is specified.
    :param rel_tol: float, optional
        Minimum relative target value change to count as improvement.
        If not provided, 0.0 will be used as default.
        Must not be provided if `abs_tol` is specified.
    :param minimize: bool, default=True
        True iff the target value must be minimized (otherwise maximized).
    """


    def __init__(
        self,
        startup=0,
        patience=1,
        max_epochs=None,
        abs_tol=None,
        rel_tol=None,
        minimize=True
    ):
        
        # Arguments

        self._startup = startup
        self._patience = patience
        self._max_epochs = max_epochs
        self._abs_tol = abs_tol
        self._rel_tol = rel_tol
        self._minimize = minimize

        # Argument check

        if (self._abs_tol is not None) and (self._rel_tol is not None):
            raise ValueError("Only `abs_tol` or `rel_tol` must be provided, not both")

        # Internal state

        self._best_target_value = None
        self._curr_epoch_num = -1
        self._curr_patience = self._patience


    def _compute_target_value(
        self,
        loss_reg_pool,
        loss_weighter
    ):
        """
        Computes the target value for early stopping.
        Calling this method must not modify the early stopper internal state.

        :param loss_reg_pool: dict of str -> gorideep.utils.loss_register.LossRegister
            Loss register pool of the training pipeline.
        :param loss_weighter: gorideep.loss_weighters.LossWeighter
            Loss weighter of the training pipeline.

        :return: float
            The computed target value.
        """
        
        raise NotImplementedError


    def _curr_target_value_improved(
        self,
        curr_target_value
    ):
        """
        Returns True iff the current target value is considered an improvement.
        
        :param curr_target_value: float
            Current target value.
        """

        target_value_thr = self._best_target_value

        if self._minimize:

            if self._abs_tol is not None:
                target_value_thr = self._best_target_value - self._abs_tol
            if self._rel_tol is not None:
                target_value_thr = self._best_target_value * (1 - self._rel_tol)
            
            return curr_target_value < target_value_thr

        else:

            if self._abs_tol is not None:
                target_value_thr = self._best_target_value + self._abs_tol
            if self._rel_tol is not None:
                target_value_thr = self._best_target_value * (1 + self._rel_tol)
            
            return curr_target_value > target_value_thr


    def update(
        self,
        loss_reg_pool,
        loss_weighter
    ):

        self._curr_epoch_num += 1

        curr_target_value = self._compute_target_value(loss_reg_pool, loss_weighter)

        if self._best_target_value is None:

            self._best_target_value = curr_target_value
        
        else:

            improvement = self._curr_target_value_improved(curr_target_value)

            if improvement:

                self._best_target_value = curr_target_value
                self._curr_patience = self._patience

            else:

                if self._curr_epoch_num > self._startup:
                    self._curr_patience -= 1        


    def early_stop(
        self
    ):
        
        if self._curr_epoch_num == 0: return False
        return (self._curr_patience == 0) or (self._curr_epoch_num >= self._max_epochs)


    def improvement(
        self
    ):
        
        if self._curr_epoch_num == 0: return False
        return self._curr_patience == self._patience


    def save(
        self,
        dirname
    ):
        
        internal_state_dict = {
            "best_target_value": self._best_target_value,
            "curr_epoch_num": self._curr_epoch_num,
            "curr_patience": self._curr_patience
        }

        internal_state_filename = os.path.join(dirname, "internal_state.json")
        goripy.file.json.save_json(internal_state_dict, internal_state_filename)


    def load(
        self,
        dirname
    ):

        internal_state_filename = os.path.join(dirname, "internal_state.json")
        internal_state_dict = goripy.file.json.load_json(internal_state_filename)

        self._best_target_value = internal_state_dict["best_target_value"]
        self._curr_epoch_num = internal_state_dict["curr_epoch_num"]
        self._curr_patience = internal_state_dict["curr_patience"]
