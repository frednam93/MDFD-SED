import torch
import torch.nn as nn
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import numpy as np
import pandas as pd
import os
import logging
import pickle
import yaml
import shutil
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from pathlib import Path


from utils.utils import *
from utils.model import *
from utils.dataset import *


def get_configs(config_dir):
    #get hyperparameters from yaml
    with open(config_dir, "r") as f:
        configs = yaml.safe_load(f)

    train_cfg = configs["training"]
    feature_cfg = configs[configs["generals"]["feature"]]
    train_cfg["net_pooling"] = feature_cfg["net_subsample"]
    return configs, train_cfg, feature_cfg


def get_save_directories(configs, train_cfg, iteration, gpu):
    general_cfg = configs["generals"]
    save_folder = general_cfg["save_folder"]
    savepsds = general_cfg["savepsds"]

    # set save folder
    if train_cfg["test_only"] and (train_cfg["test_folder"] is not None):
        save_folder = train_cfg["test_folder"]
        configs["generals"]["save_folder"] = save_folder
    else:
        if save_folder.count("new_exp") > 0:
            save_folder = save_folder + '_gpu=' + str(gpu)
            configs["generals"]["save_folder"] = save_folder
        if iteration is not None:
            save_folder_temp = save_folder + '_iter=' + str(iteration)
            # if train_cfg["test_only"] and (not os.path.isdir(save_folder)):
            #     save_folder_temp = save_folder + '_iter=0'
            save_folder = save_folder_temp
            configs["generals"]["save_folder"] = save_folder
        if not train_cfg["test_only"]:
            if os.path.isdir(save_folder):
                shutil.rmtree(save_folder)
            os.makedirs(save_folder, exist_ok=True) # saving folder
            with open(os.path.join(save_folder, 'config.yaml'), 'w') as f:
                yaml.dump(configs, f)  # save yaml in the saving folder

    #set best paths
    if train_cfg["test_only"]:
        stud_best_path = glob(os.path.join(save_folder, "best_student*.pt"))[0]
        tch_best_path = glob(os.path.join(save_folder, "best_teacher*.pt"))[0]
    else:
        stud_best_path = os.path.join(save_folder, "best_student.pt")
        tch_best_path = os.path.join(save_folder, "best_teacher.pt")
    train_cfg["best_paths"] = [stud_best_path, tch_best_path]

    # psds folder
    if savepsds:
        stud_psds_folder = os.path.join(save_folder, "psds_student")
        tch_psds_folder = os.path.join(save_folder, "psds_teacher")
        psds_folders = [stud_psds_folder, tch_psds_folder]
    else:
        psds_folders = [None, None]

    train_cfg["psds_folders"] = psds_folders
    return configs, train_cfg


def get_logger(save_folder):
    logger = logging.getLogger()
    while len(logger.handlers) > 0:                    # check if handler already exists
        logger.removeHandler(logger.handlers[0])       # Logger already exists
    formatter = logging.Formatter('%(message)s')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(save_folder, "log.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_labeldict():
    return OrderedDict({"Alarm_bell_ringing": 0,
                        "Blender": 1,
                        "Cat": 2,
                        "Dishes": 3,
                        "Dog": 4,
                        "Electric_shaver_toothbrush": 5,
                        "Frying": 6,
                        "Running_water": 7,
                        "Speech": 8,
                        "Vacuum_cleaner": 9})


def get_encoder(label_dict, feature_cfg, audio_len):
    return Encoder(list(label_dict.keys()),
                   audio_len=audio_len,
                   frame_len=feature_cfg["frame_length"],
                   frame_hop=feature_cfg["hop_length"],
                   net_pooling=feature_cfg["net_subsample"],
                   sr=feature_cfg["sr"],
                   smoothing=feature_cfg["smoothing"])


def get_mt_datasets(configs, train_cfg):
    general_cfg = configs["generals"]
    encoder = train_cfg["encoder"]
    dataset_cfg = configs["dataset"]
    batch_size_val = train_cfg["batch_size_val"]
    num_workers = train_cfg["num_workers"]
    batch_sizes = train_cfg["batch_size"]
    synthdataset_cfg = configs[general_cfg["synth_dataset"]]

    synth_train_tsv = synthdataset_cfg["synth_train_tsv"]
    synth_train_df = pd.read_csv(synth_train_tsv, sep="\t")
    weak_dir = dataset_cfg["weak_folder"]
    weak_df = pd.read_csv(dataset_cfg["weak_tsv"], sep="\t")
    weak_train_df = weak_df.sample(frac=train_cfg["weak_split"], random_state=train_cfg["seed"])
    weak_valid_df = weak_df.drop(weak_train_df.index).reset_index(drop=True)
    weak_train_df = weak_train_df.reset_index(drop=True)
    synth_valid_dir = synthdataset_cfg["synth_val_folder"]
    synth_valid_tsv = synthdataset_cfg["synth_val_tsv"]
    synth_valid_df = pd.read_csv(synth_valid_tsv, sep="\t")
    synth_valid_dur = synthdataset_cfg["synth_val_dur"]

    strong_train_dataset = StronglyLabeledDataset(synth_train_df, synthdataset_cfg["synth_train_folder"], False, encoder)
    weak_train_dataset = WeaklyLabeledDataset(weak_train_df, weak_dir, False, encoder)
    unlabeled_dataset = UnlabeledDataset(dataset_cfg["unlabeled_folder"], False, encoder)
    strong_vaild_dataset = StronglyLabeledDataset(synth_valid_df, synth_valid_dir, True, encoder)
    weak_valid_dataset = WeaklyLabeledDataset(weak_valid_df, weak_dir, True, encoder)
    test_tsv = dataset_cfg[general_cfg["test_dataset"] + "_tsv"]
    test_df = pd.read_csv(test_tsv, sep="\t")
    test_dur = dataset_cfg[general_cfg["test_dataset"] + "_dur"]
    if general_cfg["test_dataset"] in ["eval", "public", "synth_eval"]:
        mapping = dataset_cfg[general_cfg["test_dataset"] + "_mapping"]
    else:
        mapping = None
    test_dataset = StronglyLabeledDataset(test_df, dataset_cfg[general_cfg["test_dataset"] + "_folder"], True, encoder,
                                          mapping)
    if train_cfg["real_strong"]:
        real_strong_df = pd.read_csv(dataset_cfg["strong_real_tsv"], sep="\t")
        real_strong = StronglyLabeledDataset(real_strong_df, dataset_cfg["strong_real_folder"], False, encoder)
    # use portion of datasets for debugging
    if train_cfg["div_dataset"]:
        strong_train_dataset = divide_dataset(strong_train_dataset, train_cfg["div_ratio"])
        weak_train_dataset = divide_dataset(weak_train_dataset, train_cfg["div_ratio"])
        unlabeled_dataset = divide_dataset(unlabeled_dataset, train_cfg["div_ratio"])
        strong_vaild_dataset = divide_dataset(strong_vaild_dataset, train_cfg["div_ratio"])
        weak_valid_dataset = divide_dataset(weak_valid_dataset, train_cfg["div_ratio"])
        test_dataset = divide_dataset(test_dataset, train_cfg["div_ratio"])
        if train_cfg["real_strong"]:
            real_strong = divide_dataset(real_strong, train_cfg["div_ratio"])
    # build dataloaders
    if train_cfg["real_strong"]:
        strong_train_dataset = divide_dataset(strong_train_dataset, train_cfg["synth_strong_ratio"])
        strong_train_dataset = torch.utils.data.ConcatDataset([strong_train_dataset, real_strong])
    # get train dataset
    train_data = [strong_train_dataset, weak_train_dataset, unlabeled_dataset]
    train_dataset = torch.utils.data.ConcatDataset(train_data)
    train_samplers = [torch.utils.data.RandomSampler(x) for x in train_data]
    train_batch_sampler = ConcatDatasetBatchSampler(train_samplers, batch_sizes)
    train_cfg["trainloader"] = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=num_workers)
    # get valid dataset
    valid_dataset = torch.utils.data.ConcatDataset([strong_vaild_dataset, weak_valid_dataset])
    train_cfg["validloader"] = DataLoader(valid_dataset, batch_size=batch_size_val, num_workers=num_workers)
    # get test dataset
    train_cfg["testloader"] = DataLoader(test_dataset, batch_size=batch_size_val, num_workers=num_workers)
    train_cfg["train_tsvs"] = [synth_train_df, synth_train_tsv]
    train_cfg["valid_tsvs"] = [synth_valid_dir, synth_valid_tsv, synth_valid_dur, weak_dir]
    train_cfg["test_tsvs"] = [test_tsv, test_dur]
    return train_cfg


def divide_dataset(dataset, div_ratio):
    return torch.utils.data.Subset(dataset, torch.arange(int(len(dataset) / div_ratio)))


def get_models(configs, train_cfg, logger, multigpu=False):
    net = DYCRNN(**configs["DYCRNN"])

    # ema network
    ema_net = deepcopy(net)
    for param in ema_net.parameters():
        param.detach_()

    if multigpu and (train_cfg["n_gpu"] > 1):
        net = nn.DataParallel(net)
        ema_net = nn.DataParallel(ema_net)

    train_cfg["net"] = net.to(train_cfg["device"])
    train_cfg["ema_net"] = ema_net.to(train_cfg["device"])
    logger.info("Total Trainable Params: %.3f M" % (count_parameters(train_cfg["net"]) * 1e-6)) # print number of learnable parameters in the model
    return train_cfg


def get_scaler(scaler_cfg):
    return Scaler(statistic=scaler_cfg["statistic"], normtype=scaler_cfg["normtype"], dims=scaler_cfg["dims"])


def get_f1calcs(n_class, device, n_calc=2):
    if n_calc == 2:
        stud_f1calc = pl.metrics.classification.F1(n_class, average="macro", multilabel=True, compute_on_step=False)
        tch_f1calc = pl.metrics.classification.F1(n_class, average="macro", multilabel=True, compute_on_step=False)
        return stud_f1calc.to(device), tch_f1calc.to(device)
    if n_calc == 3:
        stud_f1calc = pl.metrics.classification.F1(n_class, average="macro", multilabel=True, compute_on_step=False)
        tch_f1calc = pl.metrics.classification.F1(n_class, average="macro", multilabel=True, compute_on_step=False)
        both_f1calc = pl.metrics.classification.F1(n_class, average="macro", multilabel=True, compute_on_step=False)
        return stud_f1calc.to(device), tch_f1calc.to(device), both_f1calc.to(device)


def get_printings(train_cfg):
    if train_cfg["sum_val_metric"]:
        printing_epoch = '[Epc %d] tt: %0.3f, cl_st: %0.3f, cl_wk: %0.3f, cn_st: %0.3f, cn_wk: %0.3f, ' \
                              'st_vl: %0.3f, t_vl: %0.3f, t: %ds'
    else:
        printing_epoch = '[Epc %d] tt: %0.3f, cl_st: %0.3f, cl_wk: %0.3f, cn_st: %0.3f, cn_wk: %0.3f,' \
                              ' st_wk: %0.3f, st_it: %0.3f, t_wk: %0.3f, t_it: %0.3f, t: %ds'

    printing_test = "      [student] psds1: %.4f, psds2: %.4f CB-F1: %.3f, IB-f1: %.3f"\
                    "\n      [teacher] psds1: %.4f, psds2: %.4f CB-F1: %.3f, IB-f1: %.3f"
    return printing_epoch, printing_test


class History:
    def __init__(self, train_cfg):
        self.history = {"train_total_loss": [], "train_class_strong_loss": [], "train_class_weak_loss": [],
                        "train_cons_strong_loss": [], "train_cons_weak_loss": [], "stud_val_metric": [],
                        "tch_val_metric": []}
        self.sum_val_metric = train_cfg["sum_val_metric"]

    def update(self, train_return, val_return):
        total, class_str, class_wk, cons_str, cons_wk = train_return

        if self.sum_val_metric:
            stud_val_metric, tch_val_metric = val_return
        else:
            stud_weak, stud_inter, tch_weak, tch_inter = val_return
            stud_val_metric = stud_weak + stud_inter
            tch_val_metric = tch_weak + tch_inter

        self.history['train_total_loss'].append(total)
        self.history['train_class_strong_loss'].append(class_str)
        self.history['train_class_weak_loss'].append(class_wk)
        self.history['train_cons_strong_loss'].append(cons_str)
        self.history['train_cons_weak_loss'].append(cons_wk)
        self.history['stud_val_metric'].append(stud_val_metric)
        self.history['tch_val_metric'].append(tch_val_metric)
        return stud_val_metric, tch_val_metric

    def save(self, save_dir):
        with open(save_dir, 'wb') as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)


class BestModels:
    def __init__(self, best_paths):
        self.stud_best_val_metric = 0.0
        self.tch_best_val_metric = 0.0
        self.stud_best_state_dict = None
        self.tch_best_state_dict = None
        self.best_paths = best_paths

    def update(self, train_cfg, logger, val_metrics):
        stud_update = False
        tch_update = False
        if val_metrics[0] > self.stud_best_val_metric:
            self.stud_best_val_metric = val_metrics[0]
            self.stud_best_state_dict = train_cfg["net"].state_dict()
            stud_update = True
        if val_metrics[1] > self.tch_best_val_metric:
            self.tch_best_val_metric = val_metrics[1]
            self.tch_best_state_dict = train_cfg["ema_net"].state_dict()
            tch_update = True

        if train_cfg["epoch"] > int(train_cfg["n_epochs"] * 0.5):
            if stud_update:
                if tch_update:
                    logger.info("     best student & teacher model updated at epoch %d!" % (train_cfg["epoch"] + 1))
                else:
                    logger.info("     best student model updated at epoch %d!" % (train_cfg["epoch"] + 1))
            elif tch_update:
                logger.info("     best teacher model updated at epoch %d!" % (train_cfg["epoch"] + 1))
        return logger

    def get_bests(self):
        torch.save(self.stud_best_state_dict, self.best_paths[0])
        torch.save(self.tch_best_state_dict, self.best_paths[1])
        return self.stud_best_val_metric, self.tch_best_val_metric


def rename_saved_models(save_folder, best_paths, test_returns):
    new_stud_best_name = "best_student_PSDS1=%.4f_PSDS2=%.4f_cbF1=%.3f_ibF1=%.3f.pt" % (test_returns[:4])
    new_tch_best_name = "best_teacher_PSDS1=%.4f_PSDS2=%.4f_cbF1=%.3f_ibF1=%.3f.pt" % (test_returns[4:])
    new_stud_best_path = os.path.join(save_folder, new_stud_best_name)
    new_tch_best_path = os.path.join(save_folder, new_tch_best_name)
    os.rename(best_paths[0], new_stud_best_path)
    os.rename(best_paths[1], new_tch_best_path)

