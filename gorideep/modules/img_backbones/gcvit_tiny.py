import os
import pathlib

import torch

import gorideep.external.gc_vit as ext_gc_vit



class GCVitTinyImageBackbone(torch.nn.Module):
    """
    Standard GCVit Tiny feature backbone module.
    Pre-trained weights obtained from https://github.com/NVlabs/GCViT.
    """



    class Permute(torch.nn.Module):
        """
        Wrapper module that applies torch.contiguous() to a tensor after another module.

        :param permute_dims: list of int
            Passed to torch.permute.
        """

        def __init__(self, *permute_dims):
            super().__init__()
            self.permute_dims = list(permute_dims)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.permute(x, self.permute_dims)



    def __init__(
            self
        ):
        
        super(GCVitTinyImageBackbone, self).__init__()

        # Input and output shapes

        self.img_size = 224
        self.feature_shape = (512, 7, 7)

        # Model construction

        net = ext_gc_vit.gc_vit_tiny(pretrained=False)

        self.backbone = torch.nn.Sequential(
            net.patch_embed,
            net.pos_drop,
            net.levels[0],
            net.levels[1],
            net.levels[2],
            net.levels[3],
            net.norm
        )

        self.permute = self.Permute(0, 3, 1, 2)

        # Feature shape


    def forward(self, x):
        
        x = self.backbone(x)
        x = self.permute(x)

        return x
