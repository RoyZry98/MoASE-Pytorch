import logging
import timm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from robustbench.data import load_cifar100c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import tent
import norm
import cotta_vit as cotta

from conf import cfg, load_cfg_fom_args
# from lora import inject_trainable_lora_raw
# import operators
import random
import os

logger = logging.getLogger(__name__)
load_cfg_fom_args('"CIFAR100-C evaluation.')



def evaluate(description):
    # configure model
    size = 384

    base_model = timm.create_model("vit_base_patch16_384", pretrained=False)
    base_model.head = nn.Linear(base_model.head.in_features, 100)
    path = "/data/home/zhangrongyu/code/vida/cifar/pretrained/pretrain_cifar100.t7"
    base_model = nn.DataParallel(base_model) # make parallel
    checkpoint = torch.load(path)
    base_model.load_state_dict(checkpoint['model'], strict=False)
    base_model.cuda()
    
    head_dim = 768
    

    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)
    # evaluate on each severity and type of corruption in turn
    prev_ct = "x0"
    for severity in cfg.CORRUPTION.SEVERITY:
        error_all = []
        current_process_id = os.getpid()
        logger.info(f"当前进程 ID 是: {current_process_id}")
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # continual adaptation for all corruption 
            if i_c == 0:
                try:
                    model.reset()
                    logger.info("resetting model")
                except:
                    logger.warning("not resetting model")
            else:
                logger.warning("not resetting model")
            x_test, y_test = load_cifar100c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            if size == 224:
                x_test = torch.nn.functional.interpolate(x_test, size=(224, 224), mode='bilinear', align_corners=False)
            if size == 384:
                x_test = torch.nn.functional.interpolate(x_test, size=(384, 384), mode='bilinear', align_corners=False)
            # x_test, y_test = x_test.cuda(), y_test.cuda()
            acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE, device='cuda')
            err = 1. - acc
            if i_c==0 and err>55:
                logger.info(f"\n Break: {err}")
                return
            error_all.append(err)
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
            
        print("average:", np.mean(error_all))
        # logger.info(f"average: {np.mean(error_all)}, errrr_all: {error_all}")
        logger.info(f"\n average: {np.mean(error_all)}")

def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           ap=cfg.OPTIM.AP)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model


def setup_optimizer(params):
# def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.99),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    evaluate('"CIFAR-100-C evaluation.')
