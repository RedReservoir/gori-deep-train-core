import torch



class MLPHead(torch.nn.Module):
    """
    Generic MLP head.

    The followign parameters must be lists of the same length:
        - `hidden_size_list`
        - `hidden_dropout_list`
        - `hidden_act_list`

    Supported activation functions:
        - "identity"
        - "softmax"
        - "ReLU"
        - "sigmoid"

    :param input_size: int
        Input layer size.
    :param output_size: int
        Output layer size.
    :param dropout: float, default=0
        Input layer dropout probability.
    :param act: str, default="identity"
        Output layer activation function.
    :param hidden_size_list: list of int, default=[]
        Hidden layer sizes. Empty means no hidden layers.
    :param hidden_dropout_list: list of float, default=[]
        Hidden layer dropout probabilies. Empty means no hidden layers.
    :param hidden_act_list: list of str, default=[]
        Hidden layer activation functions. Empty means no hidden layers.
    """


    def __init__(
            self,
            input_size,
            output_size,
            dropout=0,
            act="identity",
            hidden_size_list=[],
            hidden_dropout_list=[],
            hidden_act_list=[]
        ):
        
        super(MLPHead, self).__init__()


        # Param check

        hidden_list_sizes = [
            len(hidden_size_list),
            len(hidden_dropout_list),
            len(hidden_act_list)
        ]

        if not all(hidden_list_size == hidden_list_sizes[0] for hidden_list_size in hidden_list_sizes):
            raise ValueError("Hidden lists are not of equal size")


        # Build model

        input_size_list = [input_size] + hidden_size_list
        output_size_list = hidden_size_list + [output_size]
        dropout_list = hidden_dropout_list + [dropout]
        act_list = hidden_act_list + [act]

        self.layers = torch.nn.Sequential(*[
            torch.nn.Sequential(
                torch.nn.Dropout(layer_dropout) if layer_dropout != 0 else torch.nn.Identity(),
                torch.nn.Linear(layer_input_size, layer_output_size),
                self._get_activation_function_layer(layer_act)
            )
            for layer_input_size, layer_output_size, layer_dropout, layer_act in \
            zip(input_size_list, output_size_list, dropout_list, act_list)
        ])

        
    def forward(self, x):

        x = self.layers(x)

        return x


    def _get_activation_function_layer(self, act):

        if act == "identity":
            return torch.nn.Identity()
        elif act == "softmax":
            return torch.nn.Softmax(dim=1)
        elif act == "ReLU":
            return torch.nn.ReLU()
        elif act == "sigmoid":
            return torch.nn.Sigmoid()
        else:
            raise ValueError("Unknown activation function {:s}".format(act))
