from gorideep.early_stoppers.generic.base import BaseGenericEarlyStopper



class ValidationLossGenericEarlyStopper(BaseGenericEarlyStopper):
    """
    Early stopper with generic early stopping policy.
    Uses validation loss as target value.

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
    """

    def __init__(
        self,
        startup=0,
        patience=1,
        max_epochs=None,
        abs_tol=None,
        rel_tol=None
    ):
        
        # Arguments

        super().__init__(
            startup=startup,
            patience=patience,
            max_epochs=max_epochs,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            minimize=True
        )

    
    def _compute_target_value(
        self,
        loss_reg_pool,
        loss_weighter
    ):
        
        total_val_loss = 0
        
        for loss_reg_key, loss_reg in loss_reg_pool["val"].items():

            loss_value = loss_reg.epoch_total_loss_list[-1]
            loss_weight = loss_weighter.get_loss_weight(loss_reg_key)

            total_val_loss += loss_value * loss_weight
        
        return total_val_loss
