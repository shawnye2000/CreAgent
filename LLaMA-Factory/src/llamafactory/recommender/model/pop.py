

r"""
Pop
################################################

"""

import importlib
from typing import Union, List, Optional
from .base import BaseModel
import random
import torch
import torch.nn as nn


class Pop(BaseModel, nn.Module):
    r"""Pop is an fundamental model that always recommend the most popular item."""
    def __init__(self, config):
        BaseModel.__init__(self, config)
        nn.Module.__init__(self)
        self.n_items = config['max_item_num']
        self.item_cnt = torch.zeros(
            self.n_items, 1, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.max_cnt = None
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ["item_cnt", "max_cnt"]

    def forward(self):
        pass

    def calculate_loss(self, item):
        # item = interaction[self.ITEM_ID]
        item = item.to(self.item_cnt.device)
        self.item_cnt[item, :] = self.item_cnt[item, :] + 1

        self.max_cnt = torch.max(self.item_cnt, dim=0)[0]

        return torch.nn.Parameter(torch.zeros(1)).to(self.device)

    def predict(self, item):
        # item = interaction[self.ITEM_ID]
        item = item.to(self.item_cnt.device)
        result = torch.true_divide(self.item_cnt[item, :], self.max_cnt)
        return result.squeeze(-1)

    def full_sort_predict(self, interaction):
        batch_user_num = interaction[self.USER_ID].shape[0]
        result = self.item_cnt.to(torch.float64) / self.max_cnt.to(torch.float64)
        result = torch.repeat_interleave(result.unsqueeze(0), batch_user_num, dim=0)
        return result.view(-1)

    def get_full_sort_items(self, user, items):
        """Get a list of sorted items for a given user."""
        predicted_ratings = self.predict(items)
        sorted_items = self._sort_full_items(predicted_ratings, items)
        return sorted_items.tolist()

    def _sort_full_items(self, predicted_ratings, items):
        """Sort items based on their predicted ratings."""
        # Sort items based on ratings in descending order and return item indices
        _, sorted_indices = torch.sort(predicted_ratings, descending=True)
        return items[sorted_indices]