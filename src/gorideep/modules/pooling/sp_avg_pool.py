import torch



class SpatialAveragePooling(torch.nn.Module):
    """
    Performs average pooling on a feature tensor over the HW dimensions. Afterwards, flattens all
    the resulting features into a 1D embedding.

    :param pool_size: int or tuple of int
        The desired HW size after averaging and before flattenning.
    """

    def __init__(
        self,
        pool_size
    ):

        super(SpatialAveragePooling, self).__init__()
        
        self.pool = torch.nn.AdaptiveAvgPool2d(pool_size)
        self.flatten = torch.nn.Flatten(start_dim=1)


    def forward(self, x):

        x = self.pool(x)
        x = self.flatten(x)

        return x
