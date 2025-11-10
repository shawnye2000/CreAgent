import importlib
from typing import Union, List, Optional
from .base import BaseModel
import random
import torch
import torch.nn as nn

class Random(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)
        self.config = config

    def get_full_sort_items(self, users, items):
        """Get a list of sorted items for a given user."""

        sorted_items = self._sort_full_items(users, items)
        return sorted_items

    def _sort_full_items(self, user, items):
        """Return a random list of items for a given user."""
        # random_items = torch.randperm(items.size(0)).tolist()
        idx = torch.randperm(items.nelement())
        random_items = items.view(-1)[idx].view(items.size()).tolist()
        return random_items
