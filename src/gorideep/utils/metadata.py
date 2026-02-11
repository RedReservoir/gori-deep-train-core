import os
import pathlib

import numpy

import goripy.file.json
import goripy.memory.get



class CategoryMetadata:
    """
    Container class that manages category metadata.

    Metadata must consist of a list of categories in a JSON file with the following format:

    .. code-block:: text

        [
            {
                "cat_name": "<cat_name>"
            },
            ...
        ]
        
    :param cat_subset_name: str
        Name of the category subset to use.
    """

    def __init__(
        self,
        cat_subset_name
    ):

        self._cat_subset_name = cat_subset_name
        
        #

        metadata_dirname = os.path.join(os.environ["YADEEPSTYLE_DATA_HOME"], "metadata")

        self._cat_list = goripy.file.json.load_json(
            os.path.join(metadata_dirname, "{:s}_cat_list.json".format(cat_subset_name))
        )

        #

        self._cat_name_list = [
            cat["cat_name"]
            for cat in self._cat_list
        ]

        self._cat_name_to_idx_dict = {
            cat_name: cat_idx
            for cat_idx, cat_name in enumerate(self._cat_name_list)
        }


    def generate_other_to_orig_cat_idx_mapping(
        self,
        other_cat_name_list
    ):
        """
        Generates a mapping from other category indices to the original category indices from the
        metadata using category names.

        :param other_cat_name_list: list of str
            List of other category names to map from.
        
        :return: dict
            Map from the other to the original category indices.
        """
        
        other_to_orig_cat_idx_dict = {
            other_cat_idx: self._cat_name_to_idx_dict[cat_name]
            for other_cat_idx, cat_name in enumerate(other_cat_name_list)
            if cat_name in self._cat_name_to_idx_dict
        }

        return other_to_orig_cat_idx_dict


    def get_num_bytes(
        self
    ):
        """
        Computes the number of bytes that this object weights.

        :return: int
            Number of bytes that the object weights.
        """
        
        num_bytes = 0

        num_bytes += goripy.memory.get.get_obj_bytes(self._cat_list)
        num_bytes += goripy.memory.get.get_obj_bytes(self._cat_name_list)
        num_bytes += goripy.memory.get.get_obj_bytes(self._cat_name_to_idx_dict)

        return num_bytes


    @property
    def cat_list(self):
        return self._cat_list

    @property
    def cat_name_list(self):
        return self._cat_name_list

    @property
    def cat_name_to_idx_dict(self):
        return self._cat_name_to_idx_dict

    def get_num_cats(self):
        return len(self._cat_name_list)



class MultiAttributeMetadata:
    """
    Container class that manages multi-attribute metadata.

    Metadata must consist of a list of categories in a JSON file with the following format:

    .. code-block:: text

        [
            {
                "attr_name": "<attr_name>",
                "supattr_name": "<supattr_name>",
            },
            ...
        ]
            
    :param multiattr_subset_name: str
        Name of the multiattribute subset to use.
    """

    def __init__(
        self,
        multiattr_subset_name
    ):

        self._multiattr_subset_name = multiattr_subset_name

        #

        metadata_dirname = os.path.join(os.environ["YADEEPSTYLE_DATA_HOME"], "metadata")

        self._attr_list = goripy.file.json.load_json(
            os.path.join(metadata_dirname, "{:s}_multiattr_list.json".format(multiattr_subset_name))
        )

        #

        self._attr_name_list = [
            attr["attr_name"]
            for attr in self._attr_list
        ]

        self._attr_name_to_idx_dict = {
            attr_name: attr_idx
            for attr_idx, attr_name in enumerate(self._attr_name_list)
        }

        #

        self._attr_full_name_list = [
            "{:s} -> {:s}".format(attr["supattr_name"], attr["attr_name"])
            for attr in self._attr_list
        ]

        self._attr_full_name_to_idx_dict = {
            attr_full_name: attr_idx
            for attr_idx, attr_full_name in enumerate(self._attr_full_name_list)
        }

        #

        supattr_name_set = set()
        self._supattr_name_list = []
        
        for attr in self._attr_list:
            supattr_name, attr_name = attr["supattr_name"], attr["attr_name"]
            if supattr_name not in supattr_name_set:
                supattr_name_set.add(supattr_name)
                self._supattr_name_list.append(supattr_name)

        self._supattr_name_to_idx_dict = {
            supattr_name: supattr_idx
            for supattr_idx, supattr_name in enumerate(self._supattr_name_list)
        }

        #

        self._supattr_size_list = [0 for _ in range(len(self._supattr_name_list))]
        self._attr_idx_to_supattr_attr_idxs_list = []
        self._supattr_attr_idxs_to_attr_idx_list_dict = [{} for _ in range(len(self._supattr_name_list))]

        for attr_idx, attr in enumerate(self._attr_list):

            supattr_name = attr["supattr_name"]
            supattr_idx = self._supattr_name_to_idx_dict[supattr_name]

            supattr_attr_idx = self._supattr_size_list[supattr_idx]
            self._attr_idx_to_supattr_attr_idxs_list.append((supattr_idx, supattr_attr_idx))

            self._supattr_attr_idxs_to_attr_idx_list_dict[supattr_idx][supattr_attr_idx] = attr_idx

            self._supattr_size_list[supattr_idx] += 1


    def generate_other_to_orig_attr_idx_mapping(
        self,
        other_attr_full_name_list
    ):
        """
        Generates a mapping from other attribute indices to the original attribute indices from the
        metadata using attribute full names.

        Attribute full names must follow this convention: "<supattr_name> -> <attr_name>".

        :param other_attr_full_name_list: list of str
            List of other attribute full names to map from.
        
        :return: dict
            Map from the other to the original attribute indices.
        """
        
        other_to_orig_attr_idx_dict = {
            other_attr_idx: self._attr_full_name_to_idx_dict[attr_full_name]
            for other_attr_idx, attr_full_name in enumerate(other_attr_full_name_list)
            if attr_full_name in self._attr_full_name_to_idx_dict
        }

        return other_to_orig_attr_idx_dict


    def get_num_bytes(
        self
    ):
        """
        Computes the number of bytes that this object weights.

        :return: int
            Number of bytes that the object weights.
        """
        
        num_bytes = 0

        num_bytes += goripy.memory.get.get_obj_bytes(self._attr_list)
        num_bytes += goripy.memory.get.get_obj_bytes(self._attr_name_list)
        num_bytes += goripy.memory.get.get_obj_bytes(self._attr_name_to_idx_dict)
        num_bytes += goripy.memory.get.get_obj_bytes(self._attr_full_name_list)
        num_bytes += goripy.memory.get.get_obj_bytes(self._attr_full_name_to_idx_dict)
        num_bytes += goripy.memory.get.get_obj_bytes(self._supattr_name_list)
        num_bytes += goripy.memory.get.get_obj_bytes(self._supattr_name_to_idx_dict)
        num_bytes += goripy.memory.get.get_obj_bytes(self._supattr_size_list)
        num_bytes += goripy.memory.get.get_obj_bytes(self._attr_idx_to_supattr_attr_idxs_list)
        num_bytes += goripy.memory.get.get_obj_bytes(self._supattr_attr_idxs_to_attr_idx_list_dict)

        return num_bytes


    @property
    def attr_list(self):
        return self._attr_list

    @property
    def attr_name_list(self):
        return self._attr_name_list

    @property
    def attr_name_to_idx_dict(self):
        return self._attr_name_to_idx_dict

    @property
    def supattr_name_list(self):
        return self._supattr_name_list

    @property
    def supattr_name_to_idx_dict(self):
        return self._supattr_name_to_idx_dict

    @property
    def supattr_size_list(self):
        return self._supattr_size_list

    @property
    def attr_idx_to_supattr_attr_idxs_list(self):
        return self._attr_idx_to_supattr_attr_idxs_list

    @property
    def supattr_attr_idxs_to_attr_idx_list_dict(self):
        return self._supattr_attr_idxs_to_attr_idx_list_dict

    def get_num_attrs(self):
        return len(self._attr_name_list)

    def get_num_supattrs(self):
        return len(self._supattr_name_list)



class CategoryToMultiAttributeMask:
    """
    Container class that manages category to multi-attribute association metadata.

    Metadata must consist of a numpy array of:
      - Boolean data type
      - Shape (<# cats>, <# supattrs>)
    
    In each cell, `True` means the superattribute must be predicted for that category, and `False`
    means otherwise.
        
    :param cat_subset_name: str
        Name of the category subset to use.
    :param multiattr_subset_name: str
        Name of the multiattribute subset to use.
    """

    def __init__(
        self,
        cat_subset_name,
        multiattr_subset_name
    ):

        metadata_dirname = os.path.join(os.environ["YADEEPSTYLE_DATA_HOME"], "data", "metadata")

        self._cat_to_supattr_mask = numpy.load(
            os.path.join(metadata_dirname, "{:s}_cat_to_{:s}_multiattr_mask.npy".format(
                cat_subset_name, multiattr_subset_name
            ))
        )


    def get_num_bytes(
        self
    ):
        """
        Computes the number of bytes that this object weights.

        :return: int
            Number of bytes that the object weights.
        """
        
        num_bytes = 0

        num_bytes += goripy.memory.get.get_obj_bytes(self._cat_to_supattr_mask)

        return num_bytes


    @property
    def cat_to_supattr_mask(self):
        return self._cat_to_supattr_mask
