import torch
import torch.nn as nn
import torch.nn.functional as F

import model_zoo.dsnt as dsnt


class SmoothCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """

    def __init__(self, eps=0.0):
        """
        Constructor.
        Args:
            eps (float, optional): Smoothing value. Defaults to 0.
        """
        super(SmoothCrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets, w=None):
        """
        Computes the loss.
        Args:
            inputs (torch tensor [bs x n]): Predictions.
            targets (torch tensor [bs x n] or [bs]): Targets.
        Returns:
            torch tensor [bs]: Loss values.
        """
        if len(targets.size()) == 1:  # to one hot, mask -1
            mask = (targets == -1)
            targets = torch.clamp(targets, min=0)
            targets = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1).long(), 1)
        else:
            mask = (targets.sum(-1) == 0)

        if self.eps > 0:
            n_class = inputs.size(1)
            targets = targets * (1 - self.eps) + (1 - targets) * self.eps / (
                n_class - 1
            )

        loss = -targets * F.log_softmax(inputs, dim=1)

        # if len(loss.size()) == 2 and len(mask.size()) == 1:  # In case it was one-hot encoded
        #     mask = mask.unsqueeze(-1).repeat(1, loss.size(1))
        loss = loss.sum(-1).masked_fill(mask, 0)
        return loss


class SigmoidMSELoss(nn.Module):
    """
    Sigmoid on preds + MSE
    """
    def forward(self, inputs, targets):
        inputs = inputs.view(targets.size()).sigmoid()
        mask = (targets == -1)
        loss = ((inputs * 100 - targets * 100) ** 2)
        # loss = torch.abs(inputs * 100 - targets * 100)
        loss = loss.masked_fill(mask, 0)
        return loss.mean(-1)


class DSNTLoss(nn.Module):
    """
    Sigmoid on preds + MSE
    """
    def forward(self, inputs, targets):
        mask = (targets == -1).amax(-1)
        targets = targets * 2 - 1  # [0, 1] -> [-1, 1]

        heatmaps = dsnt.flat_softmax(inputs)
        coords = dsnt.dsnt(heatmaps)

        euc_losses = dsnt.euclidean_losses(coords, targets)
        reg_losses = dsnt.js_reg_losses(heatmaps, targets, sigma_t=1.0)

        loss = euc_losses + reg_losses
        loss = loss.masked_fill(mask, 0)

        return loss.mean(-1).mean(-1)


class SeriesLoss(nn.Module):
    """
    Custom loss function for series predictions.
    """
    def __init__(self, eps=0.0, weighted=False):
        """
        Constructor.

        Args:
            eps (float, optional): Smoothing factor for cross-entropy loss. Defaults to 0.0.
            weighted (bool, optional): Flag to apply class-weighted loss. Defaults to False.
        """
        super().__init__()
        self.eps = eps
        self.ce = SmoothCrossEntropyLoss(eps=eps)
        self.weighted = weighted

    def forward(self, inputs, targets):
        assert len(targets.size()) == 3

        targets = targets.view(-1, 3)  # bs * n_classes x 3

        mask = (targets.sum(-1) == 0)

        inputs = inputs.view(inputs.size(0), -1, 3).reshape(-1, 3)
        # bs x n_classes * 3 -> bs * n_classes x 3

        loss = self.ce(inputs, targets)

        if self.weighted:
            w = torch.pow(2, targets.argmax(1)).masked_fill(mask, 0)
            loss = loss * w / w.float().mean()

        return loss


class LogLoss(nn.Module):
    """
    Cross-entropy loss without softmax
    """
    def forward(self, inputs, targets):
        if len(targets.size()) == 1:  # to one hot
            targets = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1).long(), 1)
        loss = -targets * torch.log(inputs)
        return loss.sum(-1)


class StudyLoss(nn.Module):
    """
    Custom loss function for patient predictions.

    Attributes:
        eps (float): Smoothing factor for cross-entropy loss.
        weighted (bool): Flag to apply class-weighted loss.
        use_any (bool): Flag to include 'any' label in the loss calculation.
        bce (nn.BCEWithLogitsLoss): BCE loss for bowel & extravasation.
        ce (SmoothCrossEntropyLoss): CE loss for spleen, liver & kidney.
    """
    def __init__(self, eps=0.0, weighted=True, use_any=True):
        """
        Constructor.

        Args:
            eps (float, optional): Smoothing factor for cross-entropy loss. Defaults to 0.0.
            weighted (bool, optional): Flag to apply class-weighted loss. Defaults to True.
            use_any (bool, optional): Include 'any' label in the loss calculation. Defaults to True.
        """
        super().__init__()
        self.eps = eps
        self.ce = SmoothCrossEntropyLoss(eps=eps)
        # self.ce  = LogLoss()
        self.weighted = weighted
        self.use_any = use_any

    def forward(self, inputs, targets):
        """
        Forward pass for the PatientLoss class.

        Args:
            inputs (torch.Tensor): Model predictions of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: Loss value.
        """
        assert (targets.size(1) == 25) and (len(targets.size()) == 2), "Wrong target size"
        assert (inputs.size(1) == 25) and (len(inputs.size()) == 3), "Wrong input size"
        bs = targets.size(0)
        w = torch.pow(2, targets) if self.weighted else torch.ones_like(targets)

        loss = self.ce(inputs.view(-1, 3), targets.view(-1)).view(bs, -1)

        loss_scs, w_scs = loss[:, :5].flatten(), w[:, :5].flatten()
        loss_nfn, w_nfn = loss[:, 5:15].flatten(), w[:, 5:15].flatten()
        loss_ss, w_ss = loss[:, 15:].flatten(), w[:, 15:].flatten()

        loss_scs = (loss_scs * w_scs).sum() / w_scs.sum()
        loss_nfn = (loss_nfn * w_nfn).sum() / w_nfn.sum()
        loss_ss = (loss_ss * w_ss).sum() / w_ss.sum()

        if not self.use_any:
            return (loss_scs + loss_nfn + loss_ss) / 3

        any_target = targets[:, :5].amax(1)
        any_pred = inputs[:, :5].softmax(-1)[:, :, 2].amax(1)

        any_w = torch.pow(2, any_target) if self.weighted else 1
        any_target = (any_target == 2).long()

        # print(any_target, any_pred)

        any_loss = - any_target * torch.log(any_pred) - (1 - any_target) * torch.log(1 - any_pred)
        any_loss = (any_w * any_loss).sum() / any_w.sum()

        # return loss_scs, loss_nfn, loss_ss, any_loss
        return (loss_scs + loss_nfn + loss_ss + any_loss) / 4


class SpineLoss(nn.Module):
    """
    Custom loss function for the problem.

    Attributes:
        config (dict): Configuration parameters for the loss.
        device (str): Device to perform loss computations (e.g., "cuda" or "cpu").
        aux_loss_weight (float): Weight for the auxiliary loss.
        eps (float): Smoothing factor for the primary loss.
        eps_aux (float): Smoothing factor for the auxiliary loss.
        loss (torch.nn.Module): Loss function for primary predictions.
        loss_aux (torch.nn.Module): Loss function for auxiliary predictions.
    """
    def __init__(self, config, device="cuda"):
        """
        Constructor for the AbdomenLoss class.

        Args:
            config (dict): Configuration parameters for the loss.
            device (str, optional): Device to perform loss computations. Defaults to "cuda".
        """
        super().__init__()
        self.config = config
        self.device = device

        self.aux_loss_weight = config["aux_loss_weight"]
        self.eps = config.get("smoothing", 0)
        self.eps_aux = config.get("smoothing_aux", 0)

        if config["name"] == "bce":
            self.loss = nn.BCEWithLogitsLoss(reduction="none")
        elif config["name"] == "ce":
            self.loss = SmoothCrossEntropyLoss(eps=self.eps)
        elif config["name"] == "series":
            self.loss = SeriesLoss(
                eps=self.eps, weighted=config.get("weighted", False),
            )
        elif config["name"] == "study":
            self.loss = StudyLoss(
                eps=self.eps,
                weighted=config.get("weighted", False),
                use_any=config.get("use_any", False),
            )
        elif config["name"] == "sigmoid_mse":
            self.loss = SigmoidMSELoss()
        elif config["name"] == "dsnt":
            self.loss = DSNTLoss()
        else:
            raise NotImplementedError

        if config["name_aux"] == "bce":
            self.loss_aux = nn.BCEWithLogitsLoss(reduction="none")
        elif config["name_aux"] == "ce":
            self.loss_aux = SmoothCrossEntropyLoss(eps=self.eps_aux)
        elif config["name_aux"] == "study":
            self.loss_aux = StudyLoss(
                eps=self.eps,
                weighted=config.get("weighted", False),
                use_any=config.get("use_any", False),
            )
        elif config["name_aux"] == "dsnt":
            self.loss_aux = DSNTLoss()
        else:
            pass

    def prepare(self, pred, pred_aux, y, y_aux):
        """
        Prepares the predictions and targets for loss computation.

        Args:
            pred (torch.Tensor): Main predictions.
            pred_aux (list): Auxiliary predictions.
            y (torch.Tensor): Main targets.
            y_aux (list): Auxiliary targets.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Prepared predictions and targets.
        """
        if self.config["name"] == "ce":
            y = y.squeeze(1)
        elif self.config["name"] in ["bce"]:
            y = y.float()
            pred = pred.float().view(y.size())
            if self.eps:
                y = torch.clamp(y, self.eps, 1 - self.eps)
        else:
            pass

        if self.config["name_aux"] == "ce":
            y_aux = y_aux.squeeze()
        elif self.config["name_aux"] in ["bce"]:
            y_aux = y_aux.float()
            pred_aux = pred_aux.float().view(y_aux.size())
            if self.eps_aux:
                y_aux = torch.clamp(y_aux, self.eps_aux, 1 - self.eps_aux)
        else:
            pass

        return pred, pred_aux, y, y_aux

    def forward(self, pred, pred_aux, y, y_aux):
        """
        Computes the loss.

        Args:
            pred (torch.Tensor): Main predictions.
            pred_aux (list): Auxiliary predictions.
            y (torch.Tensor): Main targets.
            y_aux (list): Auxiliary targets.

        Returns:
            torch.Tensor: Loss value.
        """
        pred, pred_aux, y, y_aux = self.prepare(pred, pred_aux, y, y_aux)

        # print(pred.size())
        # print(pred_aux.size())
        # print(y.size())
        # print(y_aux.size())

        loss = self.loss(pred, y)

        if self.aux_loss_weight > 0:
            loss_aux = self.loss_aux(pred_aux, y_aux)
            loss = (1 - self.aux_loss_weight) * loss.mean() + self.aux_loss_weight * loss_aux.mean()

        return loss.mean()


class SegLoss(nn.Module):
    """
    Custom loss function for segmentation tasks.

    Attributes:
        config (dict): Configuration parameters for the loss.
        device (str): Device to perform loss computations (e.g., "cuda" or "cpu").
        aux_loss_weight (float): Weight for the auxiliary loss.
        eps (float): Smoothing factor for the primary loss.
        loss (torch.nn.Module): Loss function for primary predictions.
        loss_aux (torch.nn.Module): Loss function for auxiliary predictions.
    """
    def __init__(self, config, device="cuda"):
        """
        Constructor for the SegLoss class.

        Args:
            config (dict): Configuration parameters for the loss.
            device (str, optional): Device to perform loss computations. Defaults to "cuda".
        """
        super().__init__()
        self.config = config
        self.device = device

        self.aux_loss_weight = config["aux_loss_weight"]
        self.eps = config.get("smoothing", 0)

        if config["name"] == "bce":
            self.loss = nn.BCEWithLogitsLoss(reduction="none")
        elif config["name"] == "ce":
            self.loss = nn.CrossEntropyLoss(
                reduction="none"
            )
        else:
            raise NotImplementedError

        if config["name_aux"] == "bce":
            self.loss_aux = nn.BCEWithLogitsLoss(reduction="none")
        elif config["name_aux"] == "ce":
            self.loss_aux = nn.CrossEntropyLoss(
                reduction="none"
            )
        else:
            raise NotImplementedError

    def prepare(self, pred, pred_aux, y, y_aux):
        """
        Prepares the predictions and targets for loss computation.

        Args:
            pred (torch.Tensor): Main predictions.
            pred_aux (list): Auxiliary predictions.
            y (torch.Tensor): Main targets.
            y_aux (list): Auxiliary targets.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Prepared predictions and targets.
        """
        if self.config["name"] == "ce":
            y = y.squeeze(1).long()
        else:  # bce, lovasz, focal
            y = (
                F.one_hot(
                    y.squeeze(1).long(), num_classes=self.config["num_classes"] + 1
                )
                .permute(0, 3, 1, 2)[:, 1:]
                .float()
            )
            pred = pred.float().view(y.size())

        if self.config["name_aux"] == "ce":
            pred_aux = pred_aux.float()
            y_aux = y_aux.squeeze(1).long()
        else:
            y_aux = y_aux.float()
            pred_aux = pred_aux.float()

        if self.eps and self.config["name"] == "bce":
            y = torch.clamp(y, self.eps, 1 - self.eps)

        return pred, pred_aux, y, y_aux

    def forward(self, pred, pred_aux, y, y_aux, w=None):
        """
        Computes the loss.

        Args:
            pred (torch.Tensor): Main predictions.
            pred_aux (list): Auxiliary predictions.
            y (torch.Tensor): Main targets.
            y_aux (list): Auxiliary targets.

        Returns:
            torch.Tensor: Loss value.
        """
        pred, pred_aux, y, y_aux = self.prepare(pred, pred_aux, y, y_aux)

        loss = self.loss(pred, y)

        if w is not None:
            loss *= w
            loss = loss.sum() / w.sum()
        else:
            loss = loss.mean()

        if not self.aux_loss_weight > 0:
            return loss

        loss_aux = self.loss_aux(pred_aux, y_aux).mean()
        return (1 - self.aux_loss_weight) * loss + self.aux_loss_weight * loss_aux
