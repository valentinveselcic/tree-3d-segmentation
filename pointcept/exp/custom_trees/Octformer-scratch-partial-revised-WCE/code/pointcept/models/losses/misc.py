"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss

# Newly added loss functions for handling noisy labels and boundary-aware losses.

@LOSSES.register_module()
class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Cross-Entropy s ugrađenim zaglađivanjem oznaka (Label Smoothing).

    Ublažava overconfidence modela na kontaktnim točkama račvanja
    gdje su anotacije među različitim kutovima snimanja nekonzistentne.
    """

    def __init__(
        self,
        label_smoothing=0.1,
        reduction="mean",
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        assert 0.0 <= label_smoothing < 1.0, "label_smoothing mora biti u [0, 1)"
        assert reduction in ("mean", "sum", "none")
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # Poravnanje tenzora u oblik [N, C] i [N]
        if pred.dim() > 2:
            pred = pred.transpose(0, 1).reshape(pred.size(1), -1).transpose(0, 1)
        target = target.view(-1)

        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if target.numel() == 0:
            return pred.sum() * 0.0

        loss = F.cross_entropy(
            pred,
            target,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )
        return self.loss_weight * loss


@LOSSES.register_module()
class SymmetricCrossEntropyLoss(nn.Module):
    """Symmetric Cross Entropy (SCE = alpha * CE + beta * RCE).

    Izvor: Wang et al., 'Symmetric Cross Entropy for Robust Learning with Noisy Labels', ICCV 2019.
    RCE (Reverse Cross-Entropy) je omeđen faktorom A, što sprječava eksploziju gradijenata
    na točkama gdje je anotator povukao rez nekoliko centimetara više ili niže.
    """

    def __init__(
        self,
        alpha=1.0,
        beta=1.0,
        A=1.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.A = A
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        if pred.dim() > 2:
            pred = pred.transpose(0, 1).reshape(pred.size(1), -1).transpose(0, 1)
        target = target.view(-1)

        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if target.numel() == 0:
            return pred.sum() * 0.0

        num_classes = pred.size(1)

        # 1. Standardni Cross-Entropy (CE)
        ce_loss = F.cross_entropy(pred, target, reduction="mean")

        # 2. Reverse Cross-Entropy (RCE)
        prob = F.softmax(pred, dim=-1)
        prob = torch.clamp(prob, min=1e-7, max=1.0)

        one_hot = F.one_hot(target, num_classes=num_classes).float()
        p_c = torch.sum(one_hot * prob, dim=-1)

        # RCE zatvorena forma: A * (1 - p_c)
        rce_loss = torch.mean(self.A * (1.0 - p_c))

        total_loss = self.alpha * ce_loss + self.beta * rce_loss
        return self.loss_weight * total_loss


@LOSSES.register_module()
class BoundaryDistanceAwareLoss(nn.Module):
    """Granični gubitak utemeljen na mapi euklidske udaljenosti od spojeva klasa.

    Inspiriran Kervadec et al. (Boundary loss). Ako je točka unutar tolerancije
    (npr. 2-3 cm od račvanja), njezina penalizacija se dinamički ublažava
    kako nekonzistentni šum u ground truthu ne bi iskrivio geometrijske težine modela.
    """

    def __init__(
        self,
        margin=0.03,
        boundary_weight=0.2,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(BoundaryDistanceAwareLoss, self).__init__()
        self.margin = margin
        self.boundary_weight = boundary_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, dist_map=None, **kwargs):
        """
        Args:
            pred: Logiti modela [N, C]
            target: Indeksi klasa [N]
            dist_map: Opcionalni tenzor [N] s euklidskim udaljenostima (u metrima)
                      svake točke do najbliže točke druge klase.
        """
        if pred.dim() > 2:
            pred = pred.transpose(0, 1).reshape(pred.size(1), -1).transpose(0, 1)
        target = target.view(-1)

        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if target.numel() == 0:
            return pred.sum() * 0.0

        # Elementarni Cross-Entropy gubitak bez redukcije: [N]
        ce_loss = F.cross_entropy(pred, target, reduction="none")

        if dist_map is not None:
            dist_map = dist_map.view(-1)[valid_mask]
            # Točke unutar margine dobivaju faktor (dist / margin), minimalno boundary_weight.
            # Točke daleko od granice dobivaju puni faktor 1.0.
            distance_scale = torch.clamp(dist_map / self.margin, min=self.boundary_weight, max=1.0)
            loss = torch.mean(ce_loss * distance_scale)
        else:
            # Fallback ako dist_map nije proslijeđen u batchu
            loss = torch.mean(ce_loss)

        return self.loss_weight * loss
