import os

import numpy

from gorideep.loss_weighters.base import BaseLossWeighter



class UniformLossWeighter(BaseLossWeighter):
    """
    Assigns uniform weights to all losses.
    """


    def get_loss_weight(
        self,
        loss_reg_key
    ):

        return 1.0


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
