# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


# def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, max_norm: float = 0):
def train_one_epoch(model_s: torch.nn.Module, model_t: torch.nn.Module,
                    criterion_s: torch.nn.Module,
                    criterion_t: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    # model.train()
    model_s.train()
    model_t.train()

    criterion_s.train()
    criterion_t.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 因为torch.any不能使用none所以就是先初试了一个tensor1
        tensor1 = tensor2 = torch.zeros(300, 1)
        outputs_s, D_s, feature_s = model_s([samples, targets], tensor1, tensor2)
        # print('succeed')

        with torch.no_grad():
            outputs_t, D_t, feature_t = model_t([samples, targets], tensor1, tensor2)

        # 特征量化蒸馏
        feature_t = feature_t.to(device)
        feature_t_quan = act_quan(feature_t, 4, device)
        mseloss = torch.nn.MSELoss(reduction='mean')
        # feature_t_quan = feature_t_quan.view(2, feature_t_quan.shape[1] * feature_t_quan[2])
        # feature_s = feature_s.view(2, feature_s.shape[1] * feature_s.shape[2])
        loss_feature = mseloss(feature_t_quan, feature_s)

        # 最后的损失
        loss_dict, reshape_output1, reshape_output2 = criterion_s(outputs_s, targets)
        # 教师网路的输出需要根据学生网络修改之后的输出做匹配

        loss_dict_t, reshape_output_t1, reshape_output1_t2 = criterion_t(outputs_t, targets, reshape_output1,
                                                                         reshape_output2)
        # 现在就是根据学生的输出修改了教师网络的输出坐标reshape_output_t1和t2，但是不知道如何对应到query，这里有点疑问

        #print(reshape_output_t1, reshape_output1_t2)
        # loss_dict = criterion_s(outputs_s, targets)

        # 根据最终输出得到最后的D，要改的就是后面的outputs——>object query
        out, True_D_s, fake_feature_s2 = model_s([samples, targets], reshape_output1, reshape_output2)
        D_s = True_D_s
        with torch.no_grad():
            out_t, True_D_t, fake_feature_s2 = model_t([samples, targets], reshape_output_t1, reshape_output1_t2)
        D_t = True_D_t

        # print('已经reshape的Ds:', True_D_s)
        # # distill loss
        # distill_loss1 = torch.mean(torch.norm(D_s - D_t, p=2))
        # distill_loss2 = torch.norm(D_s - D_t, p=2)
        D_s = D_s.permute(1, 2, 0)
        D_t = D_t.permute(1, 2, 0)
        reshape_D_s = D_s.permute(0, 2, 1)
        reshape_D_t = D_t.permute(0, 2, 1)
        norm_D = torch.norm((reshape_D_s - reshape_D_t), p=2, dim=(1, 2)).unsqueeze(1)
        distill_loss3 = torch.mean(norm_D)
        # print(distill_loss1, distill_loss2, distill_loss3)

        weight_dict = criterion_s.weight_dict
        losses1 = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        losses = 0.5 * (losses1 + distill_loss3 * 3) + 0.5 * loss_feature
        print('QFD loss:', loss_feature, 'original loss:', losses1, 'distill loss:', distill_loss3, 'total loss:',
              losses)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    # y.detach()是没有梯度的，但是y_grad是有梯度的
    return y.detach() - y_grad.detach() + y_grad  # scale - scale*g + scale*g


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad  # 得到的数值是y也就是经过round函数，但是求导的是y_grad对x求导为1


def act_quan(in_feature, n_bits, device):
    in_feature = in_feature
    n_bits = n_bits
    Qn = -2 ** (n_bits - 1)
    Qp = 2 ** (n_bits - 1) - 1
    alpha = (2 * in_feature.abs().mean() / math.sqrt(Qp))
    zero_point = 0.1 * (torch.min(in_feature.detach()) - alpha * Qn)
    g = 1.0 / math.sqrt(in_feature.numel() * Qp)
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    alpha = grad_scale(alpha, g)
    alpha = torch.full((256,), alpha).to(device)
    zero_point = grad_scale(zero_point, g)
    zero_point = torch.full((256,), zero_point).to(device)
    if len(in_feature.shape) == 3:
        if in_feature.shape[0] == alpha.shape[0]:
            alpha = alpha.unsqueeze(1).unsqueeze(2)
            zero_point = zero_point.unsqueeze(1).unsqueeze(2)
        elif in_feature.shape[1] == alpha.shape[0]:
            alpha = alpha.unsqueeze(0).unsqueeze(2)
            zero_point = zero_point.unsqueeze(0).unsqueeze(2)
        elif in_feature.shape[2] == alpha.shape[0]:
            alpha = alpha.unsqueeze(0).unsqueeze(0)
            zero_point = zero_point.unsqueeze(0).unsqueeze(0)
    in_feature = round_pass((in_feature / alpha + zero_point).clamp(Qn, Qp))
    in_feature = (in_feature - zero_point) * alpha

    return in_feature


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 1000, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        tensor1 = tensor2 = torch.zeros(300, 1)
        # outputs_s, D_s, feature_s = model_s([samples, targets], tensor1, tensor2)
        outputs, D, feature = model([samples, targets], tensor1, tensor2)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = (utils.reduce_dict(loss_dict))[0]
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
