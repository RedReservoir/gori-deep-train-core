import os

import numpy

import goripy.mldl.weights

from gorideep.utils.metadata import CategoryMetadata, MultiAttributeMetadata
from gorideep.data_counters.base import BaseDataCounter



class CategoryDataCounter(BaseDataCounter):
    """
    Counts categories from a dataset.

    :param cat_subset_name: str
        Name of the category subset to use.
    """

    def __init__(
        self,
        cat_subset_name
    ):

        self._cat_subset_name = cat_subset_name

        # Create metadata object

        self._cat_metadata = CategoryMetadata(cat_subset_name)

        # Initialize counters

        self._cat_idx_count_arr = numpy.zeros(
            shape=(self._cat_metadata.get_num_cats(),),
            dtype=numpy.uint32
        )

        # Initialize weights

        self._cat_weight_arr = None


    def count(
        self,
        metadata_point
    ):

        # Reset weights

        self._cat_weight_arr = None

        # Accumulate category counts

        cat_idx_name = "{:s}_cat_idx".format(self._cat_subset_name)
        if cat_idx_name not in metadata_point: return

        self._cat_idx_count_arr[metadata_point[cat_idx_name]] += 1


    def save(
        self,
        dirname
    ):

        # Save category counts

        numpy.save(os.path.join(dirname, "cat_idx_count_arr.npy"), self._cat_idx_count_arr)

    
    def load(
        self,
        dirname
    ):

        # Reset weights

        self._cat_weight_arr = None
        
        # Load category counts

        self._cat_idx_count_arr = numpy.zeros(
            shape=(self._cat_metadata.get_num_cats(),),
            dtype=numpy.uint32
        )

        self._cat_idx_count_arr[:] = numpy.load(os.path.join(dirname, "cat_idx_count_arr.npy"))[:]


    def get_cat_weights(
        self
    ):
        """
        Computes category weights using the category counts.

        :return: numpy.ndarray
            A numpy array with the weights assigned to each category.
        """

        # If weights have been reset, re-compte them

        if self._cat_weight_arr is None:
            self._cat_weight_arr = goripy.mldl.weights.compute_freq_weights(self._cat_idx_count_arr)

        return self._cat_weight_arr



class MultiAttributeDataCounter(BaseDataCounter):
    """
    Counts multi-attributes from a dataset.

    :param multiattr_subset_name: str
        Name of the multi-attribute subset to use.
    """

    def __init__(
        self,
        multiattr_subset_name
    ):

        self._multiattr_subset_name = multiattr_subset_name

        # Create metadata object

        self._multiattr_metadata = MultiAttributeMetadata(multiattr_subset_name)

        # Initialize counters

        self._supattr_idx_count_arr_ddict = {
            supattr_name: {
                "positive": numpy.zeros(shape=(supattr_size,), dtype=numpy.uint32),
                "negative": numpy.zeros(shape=(supattr_size,), dtype=numpy.uint32)
            }
            for supattr_name, supattr_size in zip(
                self._multiattr_metadata.supattr_name_list,
                self._multiattr_metadata.supattr_size_list
            )
        }

        # Initialize weights

        self._supattr_weight_arr_ddict = None


    def count(
        self,
        metadata_point
    ):

        # Reset weights

        self._supattr_weight_arr_ddict = None

        # Accumulate attribute counts

        for supattr_name in self._multiattr_metadata.supattr_name_list:

            supattr_pos_attr_idx_arr_name = "{:s}_{:s}_pos_attr_idx_arr".format(self._multiattr_subset_name, supattr_name)
            for supattr_attr_idx in metadata_point.get(supattr_pos_attr_idx_arr_name, []):
                self._supattr_idx_count_arr_ddict[supattr_name]["positive"][supattr_attr_idx] += 1

            supattr_neg_attr_idx_arr_name = "{:s}_{:s}_neg_attr_idx_arr".format(self._multiattr_subset_name, supattr_name)
            for supattr_attr_idx in metadata_point.get(supattr_neg_attr_idx_arr_name, []):
                self._supattr_idx_count_arr_ddict[supattr_name]["negative"][supattr_attr_idx] += 1


    def save(
        self,
        dirname
    ):

        # Save attribute counts

        save_supattr_idx_count_arr_dict = {}
        for supattr_name, supattr_idx_count_arr_dict in self._supattr_idx_count_arr_ddict.items():
            for type_str, supattr_idx_count_arr in supattr_idx_count_arr_dict.items():
                save_key = "{:s};{:s}".format(type_str, supattr_name)
                save_supattr_idx_count_arr_dict[save_key] = supattr_idx_count_arr

        numpy.savez(os.path.join(dirname, "supattr_idx_count_arr_dict.npz"), **save_supattr_idx_count_arr_dict)

    
    def load(
        self,
        dirname
    ):

        # Reset weights

        self._supattr_weight_arr_ddict = None
        
        # Load attribute counts

        self._supattr_idx_count_arr_ddict = {
            supattr_name: {
                "positive": numpy.zeros(shape=(supattr_size,), dtype=numpy.uint32),
                "negative": numpy.zeros(shape=(supattr_size,), dtype=numpy.uint32)
            }
            for supattr_name, supattr_size in zip(
                self._multiattr_metadata.supattr_name_list,
                self._multiattr_metadata.supattr_size_list
            )
        }

        save_supattr_idx_count_arr_dict = numpy.load(os.path.join(dirname, "supattr_idx_count_arr_dict.npz"))

        for save_key, supattr_idx_count_arr in save_supattr_idx_count_arr_dict.items():

            save_key_tkns = save_key.split(";")
            type_str, supattr_name = save_key_tkns[0], ";".join(save_key_tkns[1:])
            
            self._supattr_idx_count_arr_ddict[supattr_name][type_str][:] = supattr_idx_count_arr[:]


    def get_attr_weights(
        self
    ):
        """
        Computes multi-attribute weights using the multi-attribute counts.
        The weights are returned with the following structure:

        ```
        {
            "<super-attribute>": {
                "positive": <positive attribute weights>,
                "negative": <negative attribute weights>
            },
            ...
        }
        ```

        :return: dict
            A dict with the weights assigned to each super-attribute and attribute.
        """

        if self._supattr_weight_arr_ddict is None:
            
            self._supattr_weight_arr_ddict = {}

            for supattr_name in self._multiattr_metadata.supattr_name_list:

                (
                    supattr_positive_weight_arr,
                    supattr_negative_weight_arr
                ) = goripy.mldl.weights.compute_pos_neg_freq_weights(
                    self._supattr_idx_count_arr_ddict[supattr_name]["positive"],
                    self._supattr_idx_count_arr_ddict[supattr_name]["negative"]
                )

                self._supattr_weight_arr_ddict[supattr_name] = {
                    "positive": supattr_positive_weight_arr,
                    "negative": supattr_negative_weight_arr
                }

        return self._supattr_weight_arr_ddict
