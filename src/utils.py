import torch
import numpy as np

from collections import defaultdict
from sklearn.metrics import roc_auc_score


def save_model(model, name, input_shape):
    sample_input = torch.ones(1, 3, *input_shape).cuda()
    traced = torch.jit.trace(model, sample_input)
    torch.jit.save(traced, name)

def evaluate_model(model, loader, loss_function):
    metrics = defaultdict(list)
    model.eval()
    for X_batch, y_batch in loader:
        preds = model(X_batch.cuda()).cpu().detach()
        y_batch = y_batch.to(torch.float)
        loss = loss_function(preds[:, 0], y_batch)
        metrics['loss'].append(loss.item())
        metrics['preds'].append(preds)
        metrics['gt'].append(y_batch)
    return np.mean(metrics['loss']), roc_auc_score(torch.cat(metrics['gt']).numpy(), torch.cat(metrics['preds']).numpy())
