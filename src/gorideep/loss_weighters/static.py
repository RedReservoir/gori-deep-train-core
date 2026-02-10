import os

import numpy

from gorideep.loss_weighters.base import BaseLossWeighter



class CustomStaticLossWeighter(BaseLossWeighter):
    """
    Assigns custom static weights to all losses.
    """


    def __init__(
        self,
        loss_reg_key_to_weight_dict
    ):

        self._loss_reg_key_to_weight_dict = loss_reg_key_to_weight_dict


    def get_loss_weight(
        self,
        loss_reg_key
    ):

        return self._loss_reg_key_to_weight_dict.get(loss_reg_key, 1.0)


    def save_epoch_data(
        self,
        dirname,
        loss_reg_key_list
    ):
        
        pass
        
        
    def save(
        self,
        dirname
    ):
        
        pass


    def load(
        self,
        dirname
    ):
        
        pass
