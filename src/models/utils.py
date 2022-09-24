import os
import random

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def fix_seed(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_confusion_matrix(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            conf_args={'normalize': None},
            heatmap_args={'cmap': 'Blues', 'annot': True, 'fmt': 'd'},
            figsize=(8, 8),
            dst=None
        ):
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, **conf_args)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, ax=ax, **heatmap_args)
    ax.set_ylabel('Label')
    ax.set_xlabel('Predict')
    if dst:
        fig.savefig(dst)
