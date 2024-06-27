import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Sampler

import numpy as np
import math
import pandas as pd
import os
import scipy
import scipy.ndimage.filters
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path
from tqdm import tqdm

from utils.model import *
from utils.evaluation_measures import compute_sed_eval_metrics
from utils.dataset import *
from utils.data_aug import *


class Encoder:
    def __init__(self, labels, audio_len, frame_len, frame_hop, net_pooling=1, sr=16000, smoothing=0):
        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()
        self.labels = labels
        self.audio_len = audio_len
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.sr = sr
        self.net_pooling = net_pooling
        n_samples = self.audio_len * self.sr
        self.n_frames = int(math.ceil(n_samples/2/self.frame_hop)*2 / self.net_pooling)
        self.smoothing = smoothing

    def _time_to_frame(self, time):
        sample = time * self.sr
        frame = sample / self.frame_hop
        return np.clip(frame / self.net_pooling, a_min=0, a_max=self.n_frames)

    def _frame_to_time(self, frame, center=False):
        if center:
            time = (frame - 0.5) * self.frame_hop / self.sr
        else:
            time = frame * self.net_pooling * self.frame_hop / self.sr
        return np.clip(time, a_min=0, a_max=self.audio_len)

    def encode_strong_df(self, events_df):
        # from event dict, generate strong label tensor sized as [n_frame, n_class]
        true_labels = self.smoothing * np.ones((self.n_frames, len(self.labels)))
        for _, row in events_df.iterrows():
            if not pd.isna(row['event_label']):
                label_idx = self.labels.index(row["event_label"])
                onset = int(self._time_to_frame(row["onset"]))           #버림 -> 해당 time frame에 걸쳐있으면 true
                offset = int(np.ceil(self._time_to_frame(row["offset"])))  #올림 -> 해당 time frame에 걸쳐있으면 true
                true_labels[onset:offset, label_idx] = 1 - self.smoothing
        return true_labels

    def encode_weak(self, events):
        # from event dict, generate weak label tensor sized as [n_class]
        labels = self.smoothing * np.ones((len(self.labels)))
        if len(events) == 0:
            return labels
        else:
            for event in events:
                labels[self.labels.index(event)] = 1 - self.smoothing
            return labels

    def decode_strong(self, outputs, expand=None):
        #from the network output sized [n_frame, n_class], generate the label/onset/offset lists
        pred = []
        for i, label_column in enumerate(outputs.T):  #outputs size = [n_class, frames]
            change_indices = self.find_contiguous_regions(label_column)
            for row in change_indices:
                onset = self._frame_to_time(row[0])
                offset = self._frame_to_time(row[1])
                if expand is not None:
                    center = (offset + onset) / 2
                    half_width = (offset - onset) / 2
                    onset = center - half_width * expand
                    offset = center + half_width * expand
                onset = np.clip(onset, a_min=0, a_max=self.audio_len)
                offset = np.clip(offset, a_min=0, a_max=self.audio_len)
                pred.append([self.labels[i], onset, offset])
        return pred


    def decode_weak(self, outputs):
        result_labels = []
        for i, value in enumerate(outputs):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def find_contiguous_regions(self, array):
        #find at which frame the label changes in the array
        change_indices = np.logical_xor(array[1:], array[:-1]).nonzero()[0]
        #shift indices to focus the frame after
        change_indices += 1
        if array[0]:
            #if first element of array is True(1), add 0 in the beggining
            #change_indices = np.append(0, change_indices)
            change_indices = np.r_[0, change_indices]
        if array[-1]:
            #if last element is True, add the length of array
            #change_indices = np.append(change_indices, array.size)
            change_indices = np.r_[change_indices, array.size]
        #reshape the result into two columns
        return change_indices.reshape((-1, 2))


def decode_pred_batch(outputs, weak_preds, filenames, encoder, thresholds, median_filter, decode_weak, pad_idx=None):
    pred_dfs = {}
    if not isinstance(median_filter, list):
        median_filter = [median_filter] * 10
    for threshold in thresholds:
        pred_dfs[threshold] = pd.DataFrame()
    for batch_idx in range(outputs.shape[0]):                                # outputs size: [bs, n_class, frames]
        for c_th in thresholds:
            output = outputs[batch_idx]                                      # output size: [n_class, frames]
            if pad_idx is not None:
                true_len = int(output.shape[-1] * pad_idx[batch_idx].item)
                output = output[:true_len]
            output = output.transpose(0, 1).detach().cpu().numpy()           # output size: [frames, n_class]
            if decode_weak:
                for class_idx in range(weak_preds.size(1)):
                    if weak_preds[batch_idx, class_idx] < c_th:
                        output[:, class_idx] = 0
            if decode_weak < 2:
                output = output > c_th
                #output = scipy.ndimage.filters.median_filter(output, (median_filter, 1))
                for class_idx in range(weak_preds.size(1)):
                    output[:, class_idx] = scipy.ndimage.filters.median_filter(output[:, class_idx], (median_filter[class_idx]))
            pred = encoder.decode_strong(output)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = Path(filenames[batch_idx]).stem + ".wav"
            pred_dfs[c_th] = pred_dfs[c_th].append(pred, ignore_index=True)
    return pred_dfs


class ConcatDatasetBatchSampler(Sampler):
    def __init__(self, samplers, batch_sizes, epoch=0):
        self.batch_sizes = batch_sizes
        self.samplers = samplers #samplers for each dataset
        self.offsets = [0] + np.cumsum([len(x) for x in self.samplers]).tolist()[:-1] #cumulative sum of sampler lengths

        self.epoch = epoch
        self.set_epoch(self.epoch)

    def _iter_one_dataset(self, c_batch_size, c_sampler, c_offset):
        batch = []
        for idx in c_sampler:
            batch.append(c_offset + idx)
            if len(batch) == c_batch_size:
                yield batch

    def set_epoch(self, epoch):
        if hasattr(self.samplers[0], "epoch"):
            for s in self.samplers:
                s.set_epoch(epoch)

    def __iter__(self):
        iterators = [iter(i) for i in self.samplers]
        tot_batch = []
        for b_num in range(len(self)): #총 batch number만큼 for loop 돌림
            for samp_idx in range(len(self.samplers)): #각 sampler의 길이만큼 for loop 돌림: [0,1,2]
                c_batch = [] #current batch list생성
                while len(c_batch) < self.batch_sizes[samp_idx]: #current sampler의 batchsize만큼 current_batch에 샘플 집어넣기
                    c_batch.append(self.offsets[samp_idx] + next(iterators[samp_idx]))
                tot_batch.extend(c_batch)
            yield tot_batch
            tot_batch = []

    def __len__(self):
        min_len = float("inf")
        for idx, sampler in enumerate(self.samplers):
            c_len = (len(sampler)) // self.batch_sizes[idx]
            min_len = min(c_len, min_len)
        return min_len #synth, weak, unlabeled dataset길이를 각각의 batch_size로 나눠서 제일 작은값을 반환


class ExponentialWarmup(object):
    def __init__(self, optimizer, max_lr, rampup_length, exponent=-5.0):
        self.optimizer = optimizer
        self.rampup_length = rampup_length
        self.max_lr = max_lr
        self.step_num = 1
        self.exponent = exponent

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _get_lr(self):
        return self.max_lr * self._get_scaling_factor()

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        self._set_lr(lr)

    # def load_state_dict(self, state_dict):
    #     self.__dict__.update(state_dict)
    #
    # def state_dict(self):
    #     return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def _get_scaling_factor(self):
        if self.rampup_length == 0:
            return 1.0
        else:
            current = np.clip(self.step_num, 0.0, self.rampup_length)
            phase = 1.0 - current / self.rampup_length
            return float(np.exp(self.exponent * phase * phase))


class CosineDecayScheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
                #param_group['lr'] = self.lr_schedule[self.iter] * 10
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


def log_sedeval_metrics(predictions, ground_truth, save_dir=None):
    """ Return the set of metrics from sed_eval
    Args:
        predictions: pd.DataFrame, the dataframe of predictions.
        ground_truth: pd.DataFrame, the dataframe of groundtruth.
        save_dir: str, path to the folder where to save the event and segment based metrics outputs.

    Returns:
        tuple, event-based macro-F1 and micro-F1, segment-based macro-F1 and micro-F1
    """
    if predictions.empty:
        return 0.0, 0.0, 0.0, 0.0

    gt = pd.read_csv(ground_truth, sep="\t")

    event_res, segment_res = compute_sed_eval_metrics(predictions, gt)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "event_f1.txt"), "w") as f:
            f.write(str(event_res))

        with open(os.path.join(save_dir, "segment_f1.txt"), "w") as f:
            f.write(str(segment_res))


    return (
        event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        event_res.results()["overall"]["f_measure"]["f_measure"],
        segment_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        segment_res.results()["overall"]["f_measure"]["f_measure"],
    )  # return also segment measures


class Scaler(nn.Module):
    def __init__(self, statistic="instance", normtype="minmax", dims=(0, 2), eps=1e-8):
        super(Scaler, self).__init__()
        self.statistic = statistic
        self.normtype = normtype
        self.dims = dims
        self.eps = eps

    def load_state_dict(self, state_dict, strict=True):
        if self.statistic == "dataset":
            super(Scaler, self).load_state_dict(state_dict, strict)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.statistic == "dataset":
            super(Scaler, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                      unexpected_keys, error_msgs)

    def forward(self, input):
        if self.statistic == "dataset":
            if self.normtype == "mean":
                return input - self.mean
            elif self.normtype == "standard":
                std = torch.sqrt(self.mean_squared - self.mean ** 2)
                return (input - self.mean) / (std + self.eps)
            else:
                raise NotImplementedError

        elif self.statistic =="instance":
            if self.normtype == "mean":
                return input - torch.mean(input, self.dims, keepdim=True)
            elif self.normtype == "standard":
                return (input - torch.mean(input, self.dims, keepdim=True)) / (
                        torch.std(input, self.dims, keepdim=True) + self.eps)
            elif self.normtype == "minmax":
                return (input - torch.amin(input, dim=self.dims, keepdim=True)) / (
                    torch.amax(input, dim=self.dims, keepdim=True)
                    - torch.amin(input, dim=self.dims, keepdim=True) + self.eps)
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError


def take_log(feature):
    amp2db = torchaudio.transforms.AmplitudeToDB(stype="amplitude")
    amp2db.amin = 1e-5
    return amp2db(feature).clamp(min=-50, max=80)


def get_masks(train_cfg, mels, labels):
    batch_size = mels.size(0)
    mask_strong = torch.zeros(batch_size).to(mels).bool()
    mask_strong[:train_cfg["batch_size"][0]] = 1  # mask_strong size = [bs]
    mask_weak = torch.zeros(batch_size).to(mels).bool()
    mask_weak[train_cfg["batch_size"][0]:(train_cfg["batch_size"][0] + train_cfg["batch_size"][1])] = 1  # mask_weak size = [bs]
    weak_labels = (torch.sum(labels[mask_weak], -1) > 0).float()  # labels_weak size = [bs, n_class] (weak data only)
    return mask_strong, mask_weak, weak_labels


def model_prediction(train_cfg, logmels_1, logmels_2):
    student_returns = train_cfg["net"](logmels_1)
    strong_pred_stud, weak_pred_stud = student_returns

    with torch.no_grad():
        teacher_returns = train_cfg["ema_net"](logmels_2)
        strong_pred_tch, weak_pred_tch = teacher_returns[0], teacher_returns[1]

    return strong_pred_stud, strong_pred_tch, weak_pred_stud, weak_pred_tch


def obtain_loss(train_cfg, model_outs, labels, weak_labels, mask_strong, mask_weak):
    strong_pred_stud, strong_pred_tch, weak_pred_stud, weak_pred_tch = model_outs
    loss_total = 0

    loss_class_weak = train_cfg["criterion_class"](weak_pred_stud[mask_weak], weak_labels)
    loss_cons_weak = train_cfg["criterion_cons"](weak_pred_stud, weak_pred_tch.detach())

    w_cons = train_cfg["w_cons_max"] * train_cfg["scheduler"]._get_scaling_factor()
    loss_class_strong = train_cfg["criterion_class"](strong_pred_stud[mask_strong], labels[mask_strong]) #strong masked label size = [bs_strong, n_class, frames]
    loss_cons_strong = train_cfg["criterion_cons"](strong_pred_stud, strong_pred_tch.detach())
    loss_total += loss_class_strong + train_cfg["w_weak"] * loss_class_weak + \
                  w_cons * (loss_cons_strong + train_cfg["w_weak_cons"] * loss_cons_weak)

    return loss_total, loss_class_strong, loss_class_weak, loss_cons_strong, loss_cons_weak


def update_ema(train_cfg):
    # update EMA model
    alpha = min(1 - 1 / train_cfg["scheduler"].step_num, train_cfg["ema_factor"])
    for ema_params, params in zip(train_cfg["ema_net"].parameters(), train_cfg["net"].parameters()):
        ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)
    return train_cfg


def lr_decay(train_cfg, logger):
    if train_cfg["step_lrdecay"]:
        if train_cfg["test_only"] + 1 == int(train_cfg["n_epochs"] / 2):
            for g in train_cfg["optimizer"].param_groups:
                g['lr'] /= 10
            logger.info('   Loss 1/10')
        elif train_cfg["test_only"] + 1 == int(train_cfg["n_epochs"] * 3 / 4):
            for g in train_cfg["optimizer"].param_groups:
                g['lr'] /= 10
            logger.info('   Loss 1/10')
    return train_cfg


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
    return total_params
