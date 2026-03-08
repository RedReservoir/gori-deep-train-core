import torch



class ModuleDictJit(torch.nn.Module):
    """
    Wrapper class that makes torch.nn.ModuleDict compatible with `torch.jit.script`.

    :param module_dict: torch.nn.ModuleDict
        A module dict to wrap.
    """

    def __init__(
        self,
        module_dict
    ):
        
        super(ModuleDictJit, self).__init__()
        
        self.module_dict = module_dict


    def forward(
        self,
        input_ten,
        module_key: str
    ):
        """
        Forward pass.

        :param input: torch.Tensor
            Input tensor to the module.
        :param module_key: str
            Key of the module to use.
        """

        for submodule_key, submodule in self.module_dict.items():
            if submodule_key == module_key: 
                return submodule(input_ten)
