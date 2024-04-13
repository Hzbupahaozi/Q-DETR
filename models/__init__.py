# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .quant_smca_detr.detr import build as build_quant
from .smca_detr.detr import build

def build_quant_model(args):
    return build_quant(args)

def build_model(args):
    return build(args)

