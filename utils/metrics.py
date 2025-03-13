import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np


def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE
