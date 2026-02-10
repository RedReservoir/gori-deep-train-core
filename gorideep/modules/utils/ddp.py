import torch



class ModuleDictDDP(torch.nn.Module):
    """
    Wrapper class that makes torch.nn.ModuleDict compatible with DDP.

    :param module_dict: torch.nn.ModuleDict
        A module dict to wrap.
    """

    def __init__(
        self,
        module_dict
    ):
        
        super(ModuleDictDDP, self).__init__()
        
        self.module_dict = module_dict


    def forward(
        self,
        input_ten,
        module_key
    ):
        """
        Forward pass.

        :param input: torch.Tensor
            Input tensor to the module.
        :param module_key: str
            Key of the module to use.
        """

        return self.module_dict[module_key](input_ten)
