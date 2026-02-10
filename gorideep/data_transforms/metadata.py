import os

import numpy
import torch

from gorideep.data_transforms.base import BaseDataTransform
from gorideep.utils.metadata import CategoryMetadata, MultiAttributeMetadata



class CategoryToProbsWeightsDataTransform(BaseDataTransform):
    """
    Converts a category to probs and weights to use for computing the BCE loss.

    Metadata must consist of a list of categories in a JSON file with the following format:

    ```
    [
        {
            "cat_name": "<cat_name>"
        },
        ...
    ]
    ```

    Expects the index of the active category in the data point. If `None` is provided rather than
    the index, this means this category does not generate loss (all weights) to zero.
        
    :param cat_subset_name: str
        Name of the category subset to use.
    :param cat_name_disable_list: list of str, default=[]
        List of category names to disable for generating loss (weight set to zero).

    :param logger: any, optional
        Logger object in case logging are needed.
    """

    def __init__(
        self,
        cat_subset_name,
        cat_name_disable_list=[],
        logger=None
    ):

        super().__init__(
            logger
        )
        
        #

        self._cat_subset_name = cat_subset_name

        # Load metadata and pre-process

        self._cat_metadata = CategoryMetadata(cat_subset_name)

        self._cat_idx_disable_list = [
            self._cat_metadata.cat_name_to_idx_dict[cat_name]
            for cat_name in cat_name_disable_list
        ]


    def __call__(
        self,
        data_point
    ):

        original_cat_idx = data_point.get("{:s}_cat_idx".format(self._cat_subset_name), None)

        if original_cat_idx is None:

            original_cat_prob_ten = torch.zeros(size=(self._cat_metadata.get_num_cats(),), dtype=torch.float)
            original_cat_weight_ten = torch.zeros(size=(self._cat_metadata.get_num_cats(),), dtype=torch.float)

        else:

            original_cat_prob_ten = torch.zeros(size=(self._cat_metadata.get_num_cats(),), dtype=torch.float)
            original_cat_prob_ten[original_cat_idx] = 1.0

            original_cat_weight_ten = torch.ones(size=(self._cat_metadata.get_num_cats(),), dtype=torch.float)
            for cat_idx in self._cat_idx_disable_list: original_cat_weight_ten[cat_idx] = 0.0

        data_point["{:s}_cat_prob_ten".format(self._cat_subset_name)] = original_cat_prob_ten
        data_point["{:s}_cat_weight_ten".format(self._cat_subset_name)] = original_cat_weight_ten

        return data_point


        
class MultiAttributeToProbsWeightsDataTransform(BaseDataTransform):
    """
    Converts multiple attributes to probs and weights to use for computing the CCE loss.

    Metadata must consist of a list of attributes in a JSON file with the following format:

    ```
    [
        {
            "attr_name": "<attr_name>",
            "supattr_name": "<supattr_name>",
        },
        ...
    ]
    ```

    Expects two arrays (for each super-attribute):
      - An array with the positive attribute indices.
      - An array with the negative attribute indices.
    Attribute indices must be zero-indices inside of each supattribute.
    If any of these arrays is not provided, or some indices are not either positive or negative,
    no loss is generated (zero weights).
    
    :param multiattr_subset_name: str
        Name of the multi-attribute subset to use.

    :param logger: any, optional
        Logger object in case logging are needed.
    """

    def __init__(
        self,
        multiattr_subset_name,
        logger=None
    ):

        super().__init__(
            logger
        )
        
        #

        self._multiattr_subset_name = multiattr_subset_name

        # Load metadata and pre-process

        self._multiattr_metadata = MultiAttributeMetadata(multiattr_subset_name)


    def __call__(
        self,
        data_point
    ):

        for supattr_name, supattr_size in zip(
            self._multiattr_metadata.supattr_name_list,
            self._multiattr_metadata.supattr_size_list
        ):

            supattr_prob_arr = numpy.zeros(shape=(supattr_size,), dtype=float)
            supattr_weight_arr = numpy.zeros(shape=(supattr_size,), dtype=float)

            #

            supattr_pos_attr_idx_arr_name = "{:s}_{:s}_pos_attr_idx_arr".format(self._multiattr_subset_name, supattr_name)
            supattr_pos_attr_idx_arr = data_point.get(supattr_pos_attr_idx_arr_name, None)

            if supattr_pos_attr_idx_arr is not None:

                supattr_prob_arr[supattr_pos_attr_idx_arr] = 1.0
                supattr_weight_arr[supattr_pos_attr_idx_arr] = 1.0

            #

            supattr_neg_attr_idx_arr_name = "{:s}_{:s}_neg_attr_idx_arr".format(self._multiattr_subset_name, supattr_name)
            supattr_neg_attr_idx_arr = data_point.get(supattr_neg_attr_idx_arr_name, None)

            if supattr_neg_attr_idx_arr is not None:

                supattr_prob_arr[supattr_neg_attr_idx_arr] = 0.0
                supattr_weight_arr[supattr_neg_attr_idx_arr] = 1.0

            #

            data_point["{:s}_{:s}_attr_prob_ten".format(self._multiattr_subset_name, supattr_name)] =\
                torch.FloatTensor(supattr_prob_arr)
            data_point["{:s}_{:s}_attr_weight_ten".format(self._multiattr_subset_name, supattr_name)] =\
                torch.FloatTensor(supattr_weight_arr)

        return data_point
