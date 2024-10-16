import torch
import numpy as np
import torch.nn as nn
from scipy.special import softmax
from sklearn.metrics import roc_auc_score


def disk_auc(truths, preds):
    """
    Compute the average auc over the 3 classes.

    Args:
        truths (np.ndarray): Ground truth labels of shape [n].
        preds (np.ndarray): Predicted probabilities of shape [n x 3].

    Returns:
        float: Average auc value.
    """
    preds = preds[truths > -1]
    truths = truths[truths > -1]

    aucs = []
    if preds.shape[-1] == 3:
        for i in range(3):
            try:
                auc = roc_auc_score(truths == i, preds[:, i])
            except ValueError:
                auc = 0.5
            aucs.append(auc)
    else:
        raise NotImplementedError

    return np.mean(aucs)


class LogLoss(nn.Module):
    """
    Logloss module.
    """
    def forward(self, inputs, targets):
        """
        Computes the log loss.

        Args:
            inputs (torch.Tensor): Predictions of shape [batch_size x num_classes].
            targets (torch.Tensor): Targets of shape [batch_size] or [batch_size x num_classes].

        Returns:
            torch.Tensor: Loss values of shape [batch_size].
        """
        if len(targets.size()) == 1:  # to one hot
            targets = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1).long(), 1)
        loss = -targets * torch.log(inputs)
        return loss.sum(-1)


def rsna_loss(truths, preds, verbose=0):
    """
    Compute the RSNA loss.

    Args:
        truths (np.ndarray): Ground truth labels of shape [batch_size x 25].
        preds (np.ndarray): Predicted probabilities of shape [batch_size x 25 x 3].
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        float: Average loss value.
        dict: Dictionary containing individual loss values for SCS, NFN, SS, and ANY.
    """
    assert truths.shape[1] == 25
    assert preds.shape[1] == 25
    assert preds.shape[2] == 3

    loss_fct = LogLoss()

    if isinstance(preds, np.ndarray):
        if preds.min() < 0:
            preds = softmax(preds, -1)

        preds = np.clip(preds, 1e-7, 1 - 1e-7)
        truths = torch.from_numpy(truths).contiguous().long()
        preds = torch.from_numpy(preds).contiguous().float()

    # Compute individual losses
    groups = [(0, 5), (5, 15), (15, 25)]
    losses = []
    for start, end in groups:
        t = truths[:, start:end].flatten(0, 1)
        p = preds[:, start:end].flatten(0, 1)
        p = p[t > -1]
        t = t[t > -1]
        w = 2**t
        loss = (loss_fct(p, t) * w).sum() / w.sum()
        losses.append(loss)

    # Any severe SCS loss
    any_pred = preds[:, :5, 2].amax(1)
    any_target = truths[:, :5].amax(1)
    any_w = 2**any_target

    any_target = (any_target == 2).long()

    if verbose:
        print(f"tgt any mean {any_target.float().mean().item():.3f}")
        print(f"pred any mean {any_pred.mean().item():.3f}")
        print()

    any_loss = -any_target * torch.log(any_pred) - (1 - any_target) * torch.log(
        1 - any_pred
    )
    any_loss = ((any_w * any_loss).sum() / any_w.sum()).item()

    # Aggregate
    loss_scs, loss_nfn, loss_ss = losses
    avg_loss = np.mean([loss_scs.item(), loss_nfn.item(), loss_ss.item(), any_loss])

    loss_dict = {
        "scs": loss_scs,
        "nfn": loss_nfn,
        "ss": loss_ss,
        "any": any_loss,
    }
    return avg_loss, loss_dict
