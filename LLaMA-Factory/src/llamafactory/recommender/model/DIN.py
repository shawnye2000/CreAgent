import torch.nn as nn
import torch

import torchsnooper

from .base import BaseModel


'''
referred to https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/din.py
'''

class DIN(BaseModel, nn.Module):
    def __init__(self, config, input_data, device):
        nn.Module.__init__(self)
        BaseModel.__init__(self, config, input_data, device)
        self.config = config
        self.item_feat_size = sum([feat.embedding_dim for feat in self.input_data_ls if feat.name.startswith('item')])
        self.user_feat_size = sum([feat.embedding_dim for feat in self.input_data_ls if feat.name.startswith('user')])


        self.total_dim_of_all_fileds = self.user_feat_size + 2 * self.item_feat_size

        self.attn = AttentionSequencePoolingLayer(embedding_dim=self.item_feat_size)
        self.fc_layer = FullyConnectedLayer(input_size=self.total_dim_of_all_fileds,
                                            hidden_unit=self.config['hid_units'],
                                            batch_norm=False,
                                            sigmoid = True,
                                            activation='relu',
                                            dropout=self.config['dropout'],
                                            dice_dim=2)

        self.loss_func = nn.BCELoss()

        self._init_weights()

    def _init_weights(self):
        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m)



    # @torchsnooper.snoop()
    def forward(self, input_data):

        user_emb, item_emb, rec_his_emb, rec_his_mask = self.embed_input_fields_for_attention_pooling(input_data)
       
        browse_atten = self.attn(item_emb.unsqueeze(dim=1),
                            rec_his_emb, rec_his_mask) 
        concat_feature = torch.cat([item_emb, browse_atten.squeeze(dim=1), user_emb], dim=-1)
        
        # fully-connected layers
        output = self.fc_layer(concat_feature).squeeze(dim=-1) #batch,1 --> batch 

        return output
        
    def train_step(self, x, y):
        loss = self.loss_func(self.predict(x), y)

        return loss
    
    def predict(self, x):

        return self.forward(x)
      
    def get_full_sort_items(self, batch_feats, items):
        """Get a list of sorted items for a given user."""
        predicted_ratings = self.predict(batch_feats)
        # predicted_ratings = torch.tensor(predicted_ratings)
        sorted_items = self._sort_full_items(predicted_ratings, items)
        return sorted_items.tolist(), predicted_ratings.tolist()

    def _sort_full_items(self, predicted_ratings, items):
        """Sort items based on their predicted ratings."""
        # Sort items based on ratings in descending order and return item indices
        _, sorted_indices = torch.sort(predicted_ratings, descending=True)
        return items[sorted_indices]


class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim=4):
        super(AttentionSequencePoolingLayer, self).__init__()

        self.local_att = LocalActivationUnit(hidden_unit=[64, 16], embedding_dim=embedding_dim, batch_norm=False)

    
    def forward(self, query_ad, user_behavior, mask=None):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # mask                : size -> batch_size * time_seq_len
        # output              : size -> batch_size * 1 * embedding_size
        
        attention_score = self.local_att(query_ad, user_behavior)
        attention_score = torch.transpose(attention_score, 1, 2)  # B * 1 * T
        
        if mask is not None:
            attention_score = attention_score.masked_fill(mask.unsqueeze(1), torch.tensor(0))
        

        # multiply weight
        output = torch.matmul(attention_score, user_behavior)

        return output
        

class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_unit=[80, 40], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4*embedding_dim,
                                       hidden_unit=hidden_unit,
                                       batch_norm=batch_norm,
                                       sigmoid=False,
                                       activation='relu',
                                       dice_dim=3)

        self.fc2 = nn.Linear(hidden_unit[-1], 1)

    # @torchsnooper.snoop()
    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(1)
        
        queries = query.expand(-1, user_behavior_len, -1)
        
        attention_input = torch.cat([queries, user_behavior, queries-user_behavior, queries*user_behavior],
             dim=-1) # as the source code, subtraction simulates verctors' difference
        
        attention_output = self.fc1(attention_input)
        attention_score = self.fc2(attention_output) # [B, T, 1]

        return attention_score


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_unit, batch_norm=False, activation='relu', sigmoid=False, dropout=None, dice_dim=None):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_unit) >= 1 
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_unit[0]))
        
        for i, h in enumerate(hidden_unit[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_unit[i]))
            
            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            # elif activation.lower() == 'dice':
            #     assert dice_dim
            #     layers.append(Dice(hidden_unit[i], dim=dice_dim))
            else:
                raise NotImplementedError

            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(hidden_unit[i], hidden_unit[i+1]))
        
        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()
        

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x) 
        
