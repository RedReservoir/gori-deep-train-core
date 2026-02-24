import os

import numpy

import goripy.mldl.weights

from gorideep.data_counters.base import BaseDataCounter



class DatasetSizeDataCounter(BaseDataCounter):
    """
    Counts number of instances in datasets.

    :param dataset_name_list: str
        List of possible dataset names.
    """

    def __init__(
        self,
        dataset_name_list
    ):

        self._dataset_name_list = dataset_name_list

        # Initialize counters

        self._dataset_name_to_idx_dict = {
            dataset_name: idx
            for idx, dataset_name in enumerate(dataset_name_list)
        }

        self._dataset_count_arr = numpy.zeros(
            shape=(len(dataset_name_list)),
            dtype=numpy.uint32
        )


    def count(
        self,
        dataset_name,
        metadata_point
    ):

        dataset_idx = self._dataset_name_to_idx_dict[dataset_name]
        self._dataset_count_arr[dataset_idx] += 1


    def save(
        self,
        dirname
    ):

        # Save dataset counts

        numpy.save(
            os.path.join(dirname, "dataset_count_arr.npy"),
            self._dataset_count_arr
        )

    
    def load(
        self,
        dirname
    ):
        
        # Load dataset counts

        self._dataset_count_arr = numpy.load(
            os.path.join(dirname, "dataset_count_arr.npy")
        )


    def get_data(
        self
    ):
        """
        Returns a copy of the dataset counter data.

        :return: list of str
            A list with all dataset names.
        :return: numpy.ndarray
            An array with the dataset counts.
        """

        return (
            self._dataset_name_list.copy(),
            self._dataset_count_arr.copy()
        )
