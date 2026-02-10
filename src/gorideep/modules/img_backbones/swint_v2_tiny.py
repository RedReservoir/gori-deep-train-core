import torch
import torchvision



class SwinTransformerV2TinyImageBackbone(torch.nn.Module):
    """
    Standard SwinT V2 Tiny feature backbone module.

    :param contiguous_after_permute: bool, default=False
        If True, all `Permute` operations will be followed by a `Contiguous` operation.
        Set this parameter to False unless there are problems with PyTorch DDP.
    """



    class LayerContiguous(torch.nn.Module):
        """
        Wrapper module that applies torch.contiguous() to a tensor after another module.
        """


        def __init__(self, prev_module):
            super().__init__()
            self.prev_module = prev_module

        def forward(self, *args, **kwargs) -> torch.Tensor:
            return self.prev_module(*args, **kwargs).contiguous()



    def __init__(
            self,
            contiguous_after_permute=False
        ):
        
        super(SwinTransformerV2TinyImageBackbone, self).__init__()

        # Input and output shapes

        self.img_size = 256
        self.feature_shape = (768, 8, 8)

        # Model construction

        weights = torchvision.models.Swin_V2_T_Weights.DEFAULT
        net = torchvision.models.swin_v2_t(weights=weights)
        
        self.features = net.features
        self.norm = net.norm
        self.permute = net.permute

        # Contiguous mods

        if contiguous_after_permute:
            self._add_contiguous_after_permute()

    
    def forward(self, x):
        
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)

        return x


    def _add_contiguous_after_permute(self):

        self.features[0][1] = self.LayerContiguous(self.features[0][1])
        self.permute = self.LayerContiguous(self.permute)
