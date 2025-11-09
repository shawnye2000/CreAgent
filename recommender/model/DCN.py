r"""
DCN
################################################
Reference:
    Ruoxi Wang at al. "Deep & Cross Network for Ad Click Predictions." in ADKDD 2017.

Reference code:
    https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/dcn.py
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

# from .DeepFM import MLPLayers
from .base import BaseModel
import torchsnooper


class MLPLayers(nn.Module):
    r"""MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(
        self,
        layers,
        dropout=0.0,
        activation="relu",
        bn=False,
        init_method=None,
        last_activation=True,
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = nn.ReLU()
            if activation_func is not None:
                mlp_modules.append(activation_func)
        if self.activation is not None and not last_activation:
            mlp_modules.pop()
        self.mlp_layers = nn.Sequential(*mlp_modules)
    #     if self.init_method is not None:
    #         self.apply(self.init_weights)

    # def init_weights(self, module):
    #     # We just initialize the module with normal distribution as the paper said
    #     if isinstance(module, nn.Linear):
    #         if self.init_method == "norm":
    #             normal_(module.weight.data, 0, 0.01)
    #         if module.bias is not None:
    #             module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)

class DCN(BaseModel, nn.Module):
    """Deep & Cross Network replaces the wide part in Wide&Deep with cross network,
    automatically construct limited high-degree cross features, and learns the corresponding weights.
    """
    def __init__(self, config, input_data):
        # super(DCN, self).__init__(init)
        nn.Module.__init__(self)
        BaseModel.__init__(self, config, input_data)
        # load parameters info
        self.mlp_hidden_size = self.config["mlp_hidden_size"]
        self.cross_layer_num = self.config["cross_layer_num"]
        self.dropout_prob = self.config["dropout_prob"]

        self.item_feat_size = sum([feat.embedding_dim for feat in self.input_data_ls if feat.name.startswith('item')])
        self.user_feat_size = sum([feat.embedding_dim for feat in self.input_data_ls if feat.name.startswith('user')])
        # self.total_dim_of_all_fileds = self.user_feat_size + 2 * self.item_feat_size

        self.total_dim_of_all_fileds = self.user_feat_size + 2 * self.item_feat_size

        # define layers and loss
        # init weight and bias of each cross layer
        self.cross_layer_w = nn.ParameterList(
            nn.Parameter(
                torch.randn(self.total_dim_of_all_fileds).to(
                    self.device
                )
            )
            for _ in range(self.cross_layer_num)
        )
        self.cross_layer_b = nn.ParameterList(
            nn.Parameter(
                torch.zeros(self.total_dim_of_all_fileds).to(
                    self.device
                )
            )
            for _ in range(self.cross_layer_num)
        )

        # size of mlp hidden layer
        size_list = [
                        self.total_dim_of_all_fileds
                    ] + self.mlp_hidden_size
        # size of cross network output
        in_feature_num = (
                self.total_dim_of_all_fileds + self.mlp_hidden_size[-1]
        )

        self.mlp_layers = MLPLayers(size_list, dropout=self.dropout_prob, bn=True)
        self.predict_layer = nn.Linear(in_feature_num, 1)
        self.sigmoid = nn.Sigmoid()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def cross_network(self, x_0):
        r"""Cross network is composed of cross layers, with each layer having the following formula.

        .. math:: x_{l+1} = x_0 {x_l^T} w_l + b_l + x_l

        :math:`x_l`, :math:`x_{l+1}` are column vectors denoting the outputs from the l -th and
        (l + 1)-th cross layers, respectively.
        :math:`w_l`, :math:`b_l` are the weight and bias parameters of the l -th layer.

        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.

        Returns:
            torch.Tensor:output of cross network, [batch_size, num_feature_field * embedding_size]

        """
        x_l = x_0
        for i in range(self.cross_layer_num):
            xl_w = torch.tensordot(x_l, self.cross_layer_w[i], dims=([1], [0]))
            xl_dot = (x_0.transpose(0, 1) * xl_w).transpose(0, 1)
            x_l = xl_dot + self.cross_layer_b[i] + x_l
        return x_l

    def forward(self, interaction):
        embedding_ls = self.embed_input_fields(interaction)

        dcn_all_embeddings = torch.cat(embedding_ls, dim=1)  # batch_size, total_dim
        # dcn_all_embeddings = self.concat_embed_input_fields(
        #     interaction
        # )  # [batch_size, num_field, embed_dim]
        # batch_size = dcn_all_embeddings.shape[0]

        # DNN
        deep_output = self.mlp_layers(dcn_all_embeddings)
        # Cross Network
        cross_output = self.cross_network(dcn_all_embeddings)
        stack = torch.cat([cross_output, deep_output], dim=-1)
        output = self.predict_layer(stack)

        return output.squeeze(1)

    def train_step(self, x, y):
        loss = self.loss_func(self.predict(x), y)

        return loss

    def predict(self, x):

        return self.sigmoid(self.forward(x))



    def get_full_sort_items(self, feats_list, items):
        """Get a list of sorted items for a given user."""
        predicted_ratings = []
        for feat in feats_list:
            predicted_ratings.append(self.predict(feat))
        predicted_ratings = torch.tensor(predicted_ratings)
        sorted_items = self._sort_full_items(predicted_ratings, items)
        return sorted_items.tolist()

    def _sort_full_items(self, predicted_ratings, items):
        """Sort items based on their predicted ratings."""
        # Sort items based on ratings in descending order and return item indices
        _, sorted_indices = torch.sort(predicted_ratings, descending=True)
        return items[sorted_indices]