"""
Evaluate Hook with Spatial Boundary Metrics

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Extended with Boundary Tolerant Metrics (Slack IoU & Boundary F1)
"""

import numpy as np
import wandb
import torch
import torch.distributed as dist
import pointops
from uuid import uuid4
from scipy.spatial import cKDTree

import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS


def compute_slack_intersection_and_union(
    coord: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    num_classes: int = 3,
    tau: float = 0.02,
    class_pair: tuple = (0, 1),
    ignore_index: int = -1,
):
    """
    Kvantifikuje presjek i uniju uz relaksaciju tolerancije tau (Slack IoU).
    Točke na spoju klasa unutar udaljenosti tau isključuju se iz penalizacije.
    """
    valid_mask = target != ignore_index
    c_val = coord[valid_mask]
    t_val = target[valid_mask]
    p_val = pred[valid_mask]

    c1, c2 = class_pair
    mask_c1 = t_val == c1
    mask_c2 = t_val == c2

    intersection = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)

    # Ako jedna od kontaktnih klasa ne postoji u sceni, standardni izračun
    if not np.any(mask_c1) or not np.any(mask_c2):
        for c in range(num_classes):
            t_c = t_val == c
            p_c = p_val == c
            intersection[c] = np.logical_and(t_c, p_c).sum()
            union[c] = np.logical_or(t_c, p_c).sum()
        return intersection, union

    tree_c1 = cKDTree(c_val[mask_c1])
    tree_c2 = cKDTree(c_val[mask_c2])

    dists_c1, _ = tree_c2.query(c_val[mask_c1], k=1, workers=-1)
    dists_c2, _ = tree_c1.query(c_val[mask_c2], k=1, workers=-1)

    relaxed_target = t_val.copy()
    idx_c1 = np.where(mask_c1)[0]
    idx_c2 = np.where(mask_c2)[0]

    # Postavljanje tampon-zone u ignore_index
    relaxed_target[idx_c1[dists_c1 <= tau]] = ignore_index
    relaxed_target[idx_c2[dists_c2 <= tau]] = ignore_index

    eval_mask = relaxed_target != ignore_index
    for c in range(num_classes):
        t_c = relaxed_target == c
        p_c = p_val == c
        intersection[c] = np.logical_and(t_c, np.logical_and(p_c, eval_mask)).sum()
        union[c] = np.logical_or(t_c, np.logical_and(p_c, eval_mask)).sum()

    return intersection, union


def compute_boundary_f1(
    coord: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    theta: float = 0.02,
    edge_width: float = 0.02,
    class_pair: tuple = (0, 1),
    ignore_index: int = -1,
):
    """
    Računa Boundary Precision, Recall i F1-score na kontaktu dviju klasa.
    """
    valid_mask = target != ignore_index
    c_val = coord[valid_mask]
    t_val = target[valid_mask]
    p_val = pred[valid_mask]

    c1, c2 = class_pair
    mask_t_c1 = t_val == c1
    mask_t_c2 = t_val == c2

    if not np.any(mask_t_c1) or not np.any(mask_t_c2):
        return 0.0, 0.0, 0.0

    tree_t_c2 = cKDTree(c_val[mask_t_c2])
    dists_t, _ = tree_t_c2.query(c_val[mask_t_c1], k=1, workers=-1)
    gt_boundary = c_val[mask_t_c1][dists_t <= edge_width]

    mask_p_c1 = p_val == c1
    mask_p_c2 = p_val == c2
    if not np.any(mask_p_c1) or not np.any(mask_p_c2):
        return 0.0, 0.0, 0.0

    tree_p_c2 = cKDTree(c_val[mask_p_c2])
    dists_p, _ = tree_p_c2.query(c_val[mask_p_c1], k=1, workers=-1)
    pred_boundary = c_val[mask_p_c1][dists_p <= edge_width]

    if len(gt_boundary) == 0 or len(pred_boundary) == 0:
        return 0.0, 0.0, 0.0

    tree_gt = cKDTree(gt_boundary)
    dists_pred_to_gt, _ = tree_gt.query(pred_boundary, k=1, workers=-1)
    prec = np.mean(dists_pred_to_gt <= theta)

    tree_pred = cKDTree(pred_boundary)
    dists_gt_to_pred, _ = tree_pred.query(gt_boundary, k=1, workers=-1)
    rec = np.mean(dists_gt_to_pred <= theta)

    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


@HOOKS.register_module()
class ClsEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>>A Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["cls_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                label,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss_avg,
                        "val/mIoU": m_iou,
                        "val/mAcc": m_acc,
                        "val/allAcc": all_acc,
                    },
                    step=wandb.run.step,
                )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_acc
        self.trainer.comm_info["current_metric_name"] = "allAcc"

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def __init__(
        self,
        write_cls_iou=False,
        eval_boundary=True,
        tau=0.02,
        theta=0.02,
        class_pair=(0, 1),
    ):
        self.write_cls_iou = write_cls_iou
        self.eval_boundary = eval_boundary
        self.tau = tau  # Tampon-zona u metrima (0.02 = 2 cm)
        self.theta = theta  # Boundary prag tolerancije (0.02 = 2 cm)
        self.class_pair = tuple(class_pair)

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>>A Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]

            if "inverse" in input_dict.keys():
                assert "origin_segment" in input_dict.keys()
                pred = pred[input_dict["inverse"]]
                segment = input_dict["origin_segment"]

            # Standardni izračun na GPU
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )

            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())

            # -------------------------------------------------------------
            # Izračun boundary metrika nad sirovom geometrijom scene
            # -------------------------------------------------------------
            if self.eval_boundary:
                if "origin_coord" in input_dict.keys():
                    coord_np = input_dict["origin_coord"].cpu().numpy()
                else:
                    coord_np = input_dict["coord"].cpu().numpy()

                target_np = segment.cpu().numpy()
                pred_np = pred.cpu().numpy()

                slack_inter, slack_uni = compute_slack_intersection_and_union(
                    coord=coord_np,
                    target=target_np,
                    pred=pred_np,
                    num_classes=self.trainer.cfg.data.num_classes,
                    tau=self.tau,
                    class_pair=self.class_pair,
                    ignore_index=self.trainer.cfg.data.ignore_index,
                )

                if comm.get_world_size() > 1:
                    t_inter = torch.tensor(slack_inter).cuda()
                    t_uni = torch.tensor(slack_uni).cuda()
                    dist.all_reduce(t_inter), dist.all_reduce(t_uni)
                    slack_inter, slack_uni = t_inter.cpu().numpy(), t_uni.cpu().numpy()

                self.trainer.storage.put_scalar("val_slack_intersection", slack_inter)
                self.trainer.storage.put_scalar("val_slack_union", slack_uni)

                _, _, b_f1 = compute_boundary_f1(
                    coord=coord_np,
                    target=target_np,
                    pred=pred_np,
                    theta=self.theta,
                    edge_width=self.tau,
                    class_pair=self.class_pair,
                    ignore_index=self.trainer.cfg.data.ignore_index,
                )
                self.trainer.storage.put_scalar("val_boundary_f1", b_f1)

            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)

        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )

        # Logiranje Boundary rezultata
        if self.eval_boundary:
            slack_inter = self.trainer.storage.history("val_slack_intersection").total
            slack_uni = self.trainer.storage.history("val_slack_union").total
            slack_iou_class = slack_inter / (slack_uni + 1e-10)
            slack_m_iou = np.mean(slack_iou_class)
            b_f1_avg = self.trainer.storage.history("val_boundary_f1").avg

            self.trainer.logger.info(
                "Boundary Eval (tau={:.2f}m, theta={:.2f}m): Slack-mIoU {:.4f} | Slack-Deblo {:.4f} | Slack-Grana {:.4f} | Boundary-F1 {:.4f}".format(
                    self.tau, self.theta, slack_m_iou, slack_iou_class[0], slack_iou_class[1], b_f1_avg
                )
            )

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            if self.eval_boundary:
                self.trainer.writer.add_scalar("val/slack_mIoU", slack_m_iou, current_epoch)
                self.trainer.writer.add_scalar("val/slack_iou_deblo", slack_iou_class[0], current_epoch)
                self.trainer.writer.add_scalar("val/boundary_f1", b_f1_avg, current_epoch)

            if self.trainer.cfg.enable_wandb:
                wandb_dict = {
                    "Epoch": current_epoch,
                    "val/loss": loss_avg,
                    "val/mIoU": m_iou,
                    "val/mAcc": m_acc,
                    "val/allAcc": all_acc,
                }
                if self.eval_boundary:
                    wandb_dict.update({
                        "val/slack_mIoU": slack_m_iou,
                        "val/slack_iou_deblo": slack_iou_class[0],
                        "val/boundary_f1": b_f1_avg,
                    })
                wandb.log(wandb_dict, step=wandb.run.step)

            if self.write_cls_iou:
                for i in range(self.trainer.cfg.data.num_classes):
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
                        iou_class[i],
                        current_epoch,
                    )
                if self.trainer.cfg.enable_wandb:
                    for i in range(self.trainer.cfg.data.num_classes):
                        wandb.log(
                            {
                                "Epoch": current_epoch,
                                f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU": iou_class[i],
                            },
                            step=wandb.run.step,
                        )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou
        self.trainer.comm_info["current_metric_name"] = "mIoU"

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self, segment_ignore_index=(-1,), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        self.valid_class_names = None
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        gt_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.trainer.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.trainer.cfg.data.names[segment_ids[i]]].append(gt_inst)

        pred_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.trainer.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue
            segment_name = self.trainer.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        for gti, gt in enumerate(gt_instances):
                            for pred in gt["matched_pred"]:
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    else:
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not cur_match[gti]:
                                hard_false_negatives += 1
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    if has_gt and has_pred:
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        thresholds, unique_indices = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        num_examples = len(y_score_sorted)
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = {}
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert len(input_dict["offset"]) == 1
            data_name = input_dict.pop("name")[0]
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]
            segment = input_dict["segment"]
            instance = input_dict["instance"]

            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                idx = idx.cpu().flatten().long()
                output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes[data_name] = dict(gt=gt_instances, pred=pred_instance)

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)

        if comm.is_main_process():
            scenes = {}
            for _ in range(len(scenes_sync)):
                r = scenes_sync.pop()
                scenes.update(r)
                del r
            scenes = list(scenes.values())
            ap_scores = self.evaluate_matches(scenes)
            all_ap = ap_scores["all_ap"]
            all_ap_50 = ap_scores["all_ap_50%"]
            all_ap_25 = ap_scores["all_ap_25%"]
            self.trainer.logger.info(
                "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                    all_ap, all_ap_50, all_ap_25
                )
            )
            for i, label_name in enumerate(self.valid_class_names):
                ap = ap_scores["classes"][label_name]["ap"]
                ap_50 = ap_scores["classes"][label_name]["ap50%"]
                ap_25 = ap_scores["classes"][label_name]["ap25%"]
                self.trainer.logger.info(
                    "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                        idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                    )
                )
            current_epoch = self.trainer.epoch + 1
            if self.trainer.writer is not None:
                self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
                self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
                self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
                self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
                if self.trainer.cfg.enable_wandb:
                    wandb.log(
                        {
                            "Epoch": current_epoch,
                            "val/loss": loss_avg,
                            "val/mAP": all_ap,
                            "val/AP50": all_ap_50,
                            "val/AP25": all_ap_25,
                        },
                        step=wandb.run.step,
                    )
            self.trainer.logger.info(
                "<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<"
            )
            self.trainer.comm_info["current_metric_value"] = all_ap_50
            self.trainer.comm_info["current_metric_name"] = "AP50"