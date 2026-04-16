import math
import sys
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import copy
from collections import OrderedDict
import torch.nn.functional as F

