import numpy
import torch

import goripy.memory.get
import goripy.memory.info



class BaseDataset(torch.utils.data.Dataset):
    """
    Base class for datasets.
    """


    def __init__(
        self
    ):

        self._data_transform = None


    def set_data_transform(
        self,
        data_transform
    ): 
        """
        Setter for the data transform to apply after data point loading.

        :param data_transform: callable
            Data transform to apply after data point loading.
        """

        self._data_transform = data_transform


    def __getitem__(
        self,
        dataset_idx
    ):
        """
        Loads one data point.

        :param dataset_idx: int
            Index of the data point in the dataset.
        
        :return: dict
            The data point.
        """

        raise NotImplementedError

    
    def getitem_metadata(
        self,
        idx
    ):
        """
        Loads one metadata point (data point with only metadata).

        :param idx: int
            Index of the data point in the dataset.
        
        :return: dict
            The metadata point.
        """

        raise NotImplementedError


    def __len__(
        self
    ):
        """
        Returns the dataset length.
        
        :return: int
            The dataset length.
        """

        raise NotImplementedError


    _split_str_to_idx_dict = {
        "train": 0, "val": 1, "test": 2
    }

    def get_split_idxs(
        self,
        split_str
    ):
        """
        Returns a subset of the dataset indices.

        Requires assigning the following attributes:
          - `_split_mask_name`
          - `_split_mask`

        :param split_str: str
            Split to which indices must belong to.
            Accepts "train", "val" or "test.
        
        :return: sequence
            The subset of dataset indices.
        """

        if self._split_mask_name is None:
            raise ValueError("Split mask not provided")

        return numpy.flatnonzero(self._split_mask == self._split_str_to_idx_dict[split_str])
    

    def get_num_bytes(
        self
    ):
        """
        Computes the total memory overhead of the data structures of the dataset.

        Requires assigning the following attributes:
          - `_byte_attr_name_list`

        :return: int
            Memory overhead in bytes.
        """

        total_num_bytes = 0

        for attr_name in self._byte_attr_name_list:
        
            attr = getattr(self, attr_name, None)
            if attr is None: continue

            total_num_bytes += goripy.memory.get.get_obj_bytes(attr)
    
        return total_num_bytes

    
    def show_num_bytes(
        self
    ):
        """
        Prints a detailed summary of the memory overhead of the data structures of the dataset.

        Requires implementing the following methods:
          - `_byte_attr_name_list`
        """
        
        max_attr_len = max(len(attr_name) for attr_name in self._byte_attr_name_list)
        max_attr_len = max(max_attr_len, len("Total"))

        line_fmt_str = "{:" + "{:d}".format(max_attr_len) + "s} : {:>11s}"

        total_num_bytes = 0

        for attr_name in self._byte_attr_name_list:
            
            attr = getattr(self, attr_name, None)
            if attr is None: continue

            num_bytes = goripy.memory.get.get_obj_bytes(attr)
            total_num_bytes += num_bytes

            num_bytes_str = goripy.memory.info.sprint_fancy_num_bytes(num_bytes)
            print_str = line_fmt_str.format(attr_name, num_bytes_str)
            print(print_str)
        
        print("-" * (max_attr_len + 13))

        total_num_bytes_str = goripy.memory.info.sprint_fancy_num_bytes(total_num_bytes)
        print_str = line_fmt_str.format("Total", total_num_bytes_str)
        print(print_str)
