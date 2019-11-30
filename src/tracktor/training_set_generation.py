import torch
import numpy
from Cython.Includes import numpy
import numpy as np


def get_random_scaling_displacement(batch_size, max_displacement, min_scale, max_scale):
    x_displacement = torch.empty(size=(batch_size, 1)).uniform_(-max_displacement, max_displacement)
    y_displacement = torch.empty(size=(batch_size, 1)).uniform_(-max_displacement, max_displacement)
    width_scaling_factor = torch.empty(size=(batch_size, 1)).uniform_(min_scale, max_scale)
    height_scaling_factor = torch.empty(size=(batch_size, 1)).uniform_(min_scale, max_scale)
    #x_displacement = torch.random.uniform(low=-1, high=1, size=batch_size)
    #y_displacement = torch.random.uniform(low=-1, high=1, size=batch_size)
    #width_scaling_factor = torch.random.uniform(low=0.5, high=2, size=batch_size)
    #height_scaling_factor = torch.random.uniform(low=0.5, high=2, size=batch_size)
    return (x_displacement, y_displacement, width_scaling_factor, height_scaling_factor)


def transform_to_xywh(gt_pos):
    gt_pos_xywh = torch.tensor([gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2] - gt_pos[0, 0], gt_pos[0, 3] - gt_pos[0, 1]])
    return gt_pos_xywh

def apply_random_factors(gt_pos_xywh, random_factors):
    batch_size = random_factors[0].size()[0]
    training_boxes_xywh = gt_pos_xywh.repeat(batch_size, 1)
    training_boxes_xywh[:, 0:1] = training_boxes_xywh[:, 0:1] + random_factors[0] * training_boxes_xywh[:, 2:3]
    training_boxes_xywh[:, 1:2] = training_boxes_xywh[:, 1:2] + random_factors[1] * training_boxes_xywh[:, 3:4]
    training_boxes_xywh[:, 2:3] = training_boxes_xywh[:, 2:3] * random_factors[2]
    training_boxes_xywh[:, 3:4] = training_boxes_xywh[:, 3:4] * random_factors[3]

    return training_boxes_xywh


def transform_to_x1y1x2y2(training_boxes_xywh):
    training_boxes = training_boxes_xywh
    training_boxes[:, 2] = training_boxes_xywh[:, 0] + training_boxes_xywh[:, 2]
    training_boxes[:, 3] = training_boxes_xywh[:, 1] + training_boxes_xywh[:, 3]
    return training_boxes


def replicate_and_randomize_boxes(gt_pos, batch_size, max_displacement=0.5, min_scale=0.5, max_scale=2):
    #gt_pos = torch.tensor([1.0, 2.0, 5.0, 7.0])
    gt_pos_xywh = transform_to_xywh(gt_pos)
    factors = get_random_scaling_displacement(batch_size, max_displacement=max_displacement, min_scale=min_scale, max_scale=max_scale)
    training_boxes_xywh = apply_random_factors(gt_pos_xywh, factors)
    return transform_to_x1y1x2y2(training_boxes_xywh)

