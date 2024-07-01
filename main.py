import torch
import torch.optim as optim
import torch.nn as nn

import os.path
import warnings
import argparse
import shutil
import matplotlib
import numpy as np
import pandas as ps
import sed_scores_eval as sse
from sed_scores_eval import intersection_based
from sed_scores_eval.base_modules.scores import create_score_dataframe
from time import time
from datetime import datetime
from tqdm import tqdm
from glob import glob
from collections import OrderedDict

from utils.dataset import *
from utils.utils import *
from utils.settings import *
from utils.data_aug import *
from utils.evaluation_measures import compute_per_intersection_macro_f1, compute_psds_from_operating_points


def main(args, iteration=None):
    matplotlib.rcParams['agg.path.chunksize'] = 10000

    # set configurations
    configs, train_cfg, feature_cfg = get_configs(args.config)
    if train_cfg["test_only"]:
        print(" "*40 + "<"*10 + "test only" + ">"*10)
    if train_cfg["debug"]:
        train_cfg["div_dataset"] = True
        train_cfg["n_epochs"] = 1
        print("!" * 10 + "   DEBUGGING MODE   " + "!" * 10)


    # set save directories
    configs, train_cfg = get_save_directories(configs, train_cfg, iteration, args.gpu)

    # set logger
    logger = get_logger(configs["generals"]["save_folder"])
    logger.info("="*50 + "start!!!!" + "="*50)

    logger.info("save directory : " + configs["generals"]["save_folder"])

    # torch information
    logger.info("date & time of start is : " + str(datetime.now()).split('.')[0])
    logger.info("torch version is: " + str(torch.__version__))
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    train_cfg["n_gpu"] = torch.cuda.device_count()
    train_cfg["multigpu"] = args.multigpu
    logger.info("number of GPUs: " + str(train_cfg["n_gpu"]))
    train_cfg["device"] = device
    logger.info("device: " + str(device))

    # seed
    torch.random.manual_seed(train_cfg["seed"])
    if device == 'cuda':
        torch.cuda.manual_seed_all(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    # do not show warning
    if not configs["generals"]["warn"]:
        warnings.filterwarnings("ignore")

    # class label dictionary
    label_dict = get_labeldict()

    # set encoder
    train_cfg["encoder"] = get_encoder(label_dict, feature_cfg, feature_cfg["audio_max_len"])

    # set Dataloaders
    train_cfg = get_mt_datasets(configs, train_cfg)

    # set network
    train_cfg = get_models(configs, train_cfg, logger, args.multigpu)

    # set feature
    train_cfg["feat_ext"] = setmelspectrogram(feature_cfg).to(device)

    # set scaler
    train_cfg["scaler"] = get_scaler(configs["scaler"])
    printing_epoch, printing_test = get_printings(train_cfg)


    ##############################                   TRAIN / VALIDATION                   ##############################
    if not train_cfg["test_only"]:
        logger.info('   training starts!')
        start_time = time()

        #### training setup
        # get f1 calculators
        train_cfg["f1calcs"] = get_f1calcs(len(label_dict), device)
        # get optimizer, scheduler
        train_cfg["criterion_class"] = nn.BCELoss().to(device)
        if train_cfg["consistency_loss"] == "MSE":
            train_cfg["criterion_cons"] = nn.MSELoss().to(device)
        elif train_cfg["consistency_loss"] == "BCE":
            train_cfg["criterion_cons"] = nn.BCELoss().to(device)
        train_cfg["optimizer"] = optim.Adam(train_cfg["net"].parameters(), 1e-3, betas=(0.9, 0.999))
        warmup_steps = train_cfg["n_epochs_warmup"] * len(train_cfg["trainloader"])
        train_cfg["scheduler"] = ExponentialWarmup(train_cfg["optimizer"], configs["opt"]["lr"], warmup_steps)
        # etc
        history = History(train_cfg)
        bestmodels = BestModels(train_cfg["best_paths"])

        #### train!
        for train_cfg["epoch"] in range(train_cfg["n_epochs"]):
            epoch_time = time()

            #training
            train_return = train_mt(train_cfg)
            val_return = validation(train_cfg)

            #save best model when best validation metrics occur
            val_metrics = history.update(train_return, val_return)
            logger.info(printing_epoch % ((train_cfg["epoch"] + 1,) + train_return + val_return + (time() - epoch_time,)))
            logger = bestmodels.update(train_cfg, logger, val_metrics)

            #reduce lr when val_loss do not improve for 5 epochs
            train_cfg = lr_decay(train_cfg, logger)

        #save model parameters & history dictionary
        logger.info("   training took %.2f mins" % ((time()-start_time)/60))
        logger.info("        best student/teacher val_metrics: %.3f / %.3f" % bestmodels.get_bests())
        history.save(os.path.join(configs["generals"]["save_folder"], "history.pickle"))


    ##############################                          TEST                          ##############################
    # test on best model
    train_cfg["net"].load_state_dict(torch.load(train_cfg["best_paths"][0]))
    train_cfg["ema_net"].load_state_dict(torch.load(train_cfg["best_paths"][1]))

    test_returns = test(train_cfg)
    logger.info(printing_test % test_returns)
    if not train_cfg["test_only"]:
        rename_saved_models(configs["generals"]["save_folder"], train_cfg["best_paths"], test_returns)

    # delete debugging folder
    if train_cfg["debug"] and not train_cfg["test_only"]:
        shutil.rmtree(configs["generals"]["save_folder"])

    logger.info("date & time of end is : " + str(datetime.now()).split('.')[0])
    logger.info("<"*30 + "DONE!" + ">"*30)
    logging.shutdown()


########################################################################################################################
#                                                 TRAIN - Mean Teacher                                                 #
########################################################################################################################
def train_mt(train_cfg):
    train_cfg["net"].train()
    train_cfg["ema_net"].train()
    total_loss, class_strong_loss, class_weak_loss, cons_strong_loss, cons_weak_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    n_train = len(train_cfg["trainloader"])
    tk0 = tqdm(train_cfg["trainloader"], total=n_train, leave=False, desc="training processing")
    for _, (wavs, labels, _, _) in enumerate(tk0, 0):
        wavs, labels = wavs.to(train_cfg["device"]), labels.to(train_cfg["device"])    # labels size = [bs, n_class, frames]
        mels = train_cfg["feat_ext"](wavs)                                             # mels size = [bs, freqs, frames]

        # generate masks and weak labels
        mask_strong, mask_weak, weak_labels = get_masks(train_cfg, mels, labels)

        #apply data augmentations
        mels_1, mels_2, labels, weak_labels = data_augs(train_cfg, wavs, mels, labels, weak_labels, mask_strong, mask_weak)

        # log & scale
        logmels_1 = train_cfg["scaler"](take_log(mels_1))
        logmels_2 = train_cfg["scaler"](take_log(mels_2))

        # model predictions
        train_cfg["optimizer"].zero_grad()
        model_outs = model_prediction(train_cfg, logmels_1, logmels_2)

        # loss functions
        loss_total, loss_class_strong, loss_class_weak, loss_cons_strong, loss_cons_weak = \
            obtain_loss(train_cfg, model_outs, labels, weak_labels, mask_strong, mask_weak)

        # update student model
        loss_total.backward()
        train_cfg["optimizer"].step()
        train_cfg["scheduler"].step()

        # update teacher model
        train_cfg = update_ema(train_cfg)

        # update loss
        total_loss += loss_total.item()
        class_strong_loss += loss_class_strong.item()
        class_weak_loss += loss_class_weak.item()
        cons_strong_loss += loss_cons_strong.item()
        cons_weak_loss = loss_cons_weak.item()

    total_loss /= n_train
    class_strong_loss /= n_train
    class_weak_loss /= n_train
    cons_strong_loss /= n_train
    cons_weak_loss /= n_train

    return total_loss, class_strong_loss, class_weak_loss, cons_strong_loss, cons_weak_loss


########################################################################################################################
#                                                      VALIDATION                                                      #
########################################################################################################################
def validation(train_cfg):
    encoder = train_cfg["encoder"]
    train_cfg["net"].eval()
    train_cfg["ema_net"].eval()
    n_valid = len(train_cfg["validloader"])
    for f1calc in train_cfg["f1calcs"]:
        f1calc.reset()
    val_stud_buffer = {k: pd.DataFrame() for k in train_cfg["val_thresholds"]}
    val_tch_buffer = {k: pd.DataFrame() for k in train_cfg["val_thresholds"]}
    synth_valid_dir, synth_valid_tsv, synth_valid_dur, weak_dir = train_cfg["valid_tsvs"]
    decode_kwargs = {"encoder": encoder, "median_filter": train_cfg["median_window"],
                     "decode_weak": train_cfg["decode_weak_valid"]}
    with torch.no_grad():
        tk1 = tqdm(train_cfg["validloader"], total=n_valid, leave=False, desc="validation processing")
        for _, (wavs, labels, _, indexes, filenames, paths) in enumerate(tk1, 0):
            wavs, labels = wavs.to(train_cfg["device"]), labels.to(train_cfg["device"]) # labels size = [bs, n_class, frames]
            mels = train_cfg["feat_ext"](wavs)  # mels size = [bs, freqs, frames]

            logmels = train_cfg["scaler"](take_log(mels))
            model_outs = model_prediction(train_cfg, logmels, logmels)
            strong_pred_stud, strong_pred_tch, weak_pred_stud, weak_pred_tch = model_outs[:4] #ignore freq_pred

            mask_weak = torch.ones(labels.size(0)).to(logmels).bool()
            mask_strong = torch.zeros(labels.size(0)).to(logmels).bool()

            if torch.any(mask_weak):
                labels_weak = (torch.sum(labels[mask_weak], -1) > 0).float()  # labels_weak size = [bs, n_class]
                #accumulate f1score for weak labels
                train_cfg["f1calcs"][0](weak_pred_stud[mask_weak], labels_weak)
                train_cfg["f1calcs"][1](weak_pred_tch[mask_weak], labels_weak)

            if torch.any(mask_strong):
                #decoded_stud/tch_strong for intersection f1 score
                paths_strong = [x for x in paths if Path(x).parent == Path(synth_valid_dir)]
                stud_pred_dfs = decode_pred_batch(strong_pred_stud[mask_strong], weak_pred_stud[mask_strong],
                                                  paths_strong, thresholds=list(val_stud_buffer.keys()), **decode_kwargs)
                tch_pred_dfs = decode_pred_batch(strong_pred_tch[mask_strong], weak_pred_tch[mask_strong],
                                                 paths_strong, thresholds=list(val_tch_buffer.keys()), **decode_kwargs)
                for th in val_stud_buffer.keys():
                    val_stud_buffer[th] = val_stud_buffer[th].append(stud_pred_dfs[th], ignore_index=True)
                for th in val_tch_buffer.keys():
                    val_tch_buffer[th] = val_tch_buffer[th].append(tch_pred_dfs[th], ignore_index=True)


    stud_weak_f1 = train_cfg["f1calcs"][0].compute()
    tch_weak_f1 = train_cfg["f1calcs"][1].compute()
    stud_intersection_f1 = compute_per_intersection_macro_f1(val_stud_buffer, synth_valid_tsv, synth_valid_dur)
    tch_intersection_f1 = compute_per_intersection_macro_f1(val_tch_buffer, synth_valid_tsv, synth_valid_dur)
    if train_cfg["sum_val_metric"]:
        stud_val_metric = stud_weak_f1.item() * train_cfg["weakf1_ratio"] + stud_intersection_f1
        tch_val_metric = tch_weak_f1.item() * train_cfg["weakf1_ratio"] + tch_intersection_f1
        return stud_val_metric, tch_val_metric
    else:
        return stud_weak_f1.item(), stud_intersection_f1, tch_weak_f1.item(), tch_intersection_f1


########################################################################################################################
#                                                         TEST                                                         #
########################################################################################################################
def test(train_cfg):
    encoder = train_cfg["encoder"]
    psds_folders = train_cfg["psds_folders"]
    train_cfg["net"].eval()
    train_cfg["ema_net"].eval()
    test_tsv, test_dur = train_cfg["test_tsvs"]
    thresholds = np.arange(1 / (train_cfg["n_test_thresholds"] * 2), 1, 1 / train_cfg["n_test_thresholds"])
    if train_cfg["save_score"]:
        test_buffer_unprocessed = {}
    elif not train_cfg["true_psds"]:
        stud_test_psds_buffer = {k: pd.DataFrame() for k in thresholds}
        tch_test_psds_buffer = {k: pd.DataFrame() for k in thresholds}
    else:
        stud_scores = {}
        tch_scores = {}
        timestamps = encoder._frame_to_time(np.arange(157))
        event_classes = encoder.labels
    stud_test_f1_buffer = pd.DataFrame()
    tch_test_f1_buffer = pd.DataFrame()
    decode_kwargs = {"encoder": encoder, "median_filter": train_cfg["median_window"],
                     "decode_weak": train_cfg["decode_weak_test"]}
    tk2 = tqdm(train_cfg["testloader"], total=len(train_cfg["testloader"]), leave=False, desc="test processing")
    with torch.no_grad():
        for _, (wavs, labels, _, indexes, filenames, paths) in enumerate(tk2, 0):
            wavs, labels = wavs.to(train_cfg["device"]), labels.to(train_cfg["device"]) # labels size = [bs, n_class, frames]
            mels = train_cfg["feat_ext"](wavs)  # mels size = [bs, freqs, frames]

            logmels = train_cfg["scaler"](take_log(mels))
            model_outs = model_prediction(train_cfg, logmels, logmels)
            strong_pred_stud, strong_pred_tch, weak_pred_stud, weak_pred_tch = model_outs[:5] #ignore freq_pred

            # get F1 scores
            stud_pred_df_halfpoint = decode_pred_batch(strong_pred_stud, weak_pred_stud, paths, thresholds=[0.5],
                                                       **decode_kwargs)
            tch_pred_df_halfpoint = decode_pred_batch(strong_pred_tch, weak_pred_tch, paths, thresholds=[0.5],
                                                      **decode_kwargs)
            stud_test_f1_buffer = stud_test_f1_buffer.append(stud_pred_df_halfpoint[0.5], ignore_index=True)
            tch_test_f1_buffer = tch_test_f1_buffer.append(tch_pred_df_halfpoint[0.5], ignore_index=True)

            # get PSDS

            if train_cfg["save_score"]: #applied on teacher only
                scores_raw = {}
                for j in range(strong_pred_tch.shape[0]):
                    audio_id = Path(filenames[j]).stem
                    c_scores = strong_pred_tch[j]
                    c_scores = c_scores.transpose(0, 1).detach().cpu().numpy()
                    scores_raw[audio_id] = create_score_dataframe(
                        scores=c_scores,
                        timestamps=encoder._frame_to_time(np.arange(len(c_scores) + 1)),
                        event_classes=encoder.labels,)
                test_buffer_unprocessed.update(scores_raw)

            elif not train_cfg["true_psds"]:
                stud_pred_dfs = decode_pred_batch(strong_pred_stud, weak_pred_stud, paths,
                                                  thresholds=list(stud_test_psds_buffer.keys()), **decode_kwargs)
                tch_pred_dfs = decode_pred_batch(strong_pred_tch, weak_pred_tch, paths,
                                                 thresholds=list(tch_test_psds_buffer.keys()), **decode_kwargs)
                for th in stud_test_psds_buffer.keys():
                    stud_test_psds_buffer[th] = stud_test_psds_buffer[th].append(stud_pred_dfs[th], ignore_index=True)
                for th in tch_test_psds_buffer.keys():
                    tch_test_psds_buffer[th] = tch_test_psds_buffer[th].append(tch_pred_dfs[th], ignore_index=True)
            else:
                if isinstance(train_cfg["median_window"], int):
                    train_cfg["median_window"] = [train_cfg["median_window"]] * 10
                for i, filename in enumerate(filenames):
                    stud_pred = strong_pred_stud[i].transpose(0, 1).cpu().numpy()
                    tch_pred = strong_pred_tch[i].transpose(0, 1).cpu().numpy()
                    for class_idx in range(10):
                        stud_pred[:, class_idx] = scipy.ndimage.filters.median_filter(stud_pred[:, class_idx],
                                                                                      (train_cfg["median_window"][class_idx]))
                        tch_pred[:, class_idx] = scipy.ndimage.filters.median_filter(tch_pred[:, class_idx],
                                                                                     (train_cfg["median_window"][class_idx]))
                    # weak masking
                    stud_pred_weak = weak_pred_stud[i].cpu().numpy()
                    tch_pred_weak = weak_pred_tch[i].cpu().numpy()
                    for class_idx in range(10):
                        stud_pred[:, class_idx][stud_pred[:, class_idx] > stud_pred_weak[class_idx]] = \
                            stud_pred_weak[class_idx]
                        tch_pred[:, class_idx][tch_pred[:, class_idx] > tch_pred_weak[class_idx]] = \
                            tch_pred_weak[class_idx]

                    filename = filename.replace('.wav', '')
                    stud_scores[filename] = sse.utils.create_score_dataframe(stud_pred, timestamps, event_classes)
                    tch_scores[filename] = sse.utils.create_score_dataframe(tch_pred, timestamps, event_classes)


    # calculate psds
    if train_cfg["save_score"]:
        save_dir_unprocessed = os.path.join(train_cfg["test_folder"], "unprocessed")
        if not os.path.exists(save_dir_unprocessed):
            os.makedirs(save_dir_unprocessed)
        sse.io.write_sed_scores(test_buffer_unprocessed, save_dir_unprocessed)
        stud_psds1, stud_psds2, stud_CB_f1, stud_IB_f1, tch_psds1, tch_psds2, tch_CB_f1, tch_IB_f1 = 0, 0, 0, 0, 0, 0, 0, 0,
    elif not train_cfg["true_psds"]:
        psds1_kwargs = {"dtc_threshold": 0.7, "gtc_threshold": 0.7, "alpha_ct": 0, "alpha_st": 1}
        psds2_kwargs = {"dtc_threshold": 0.1, "gtc_threshold": 0.1, "cttc_threshold": 0.3, "alpha_ct": 0.5,
                        "alpha_st": 1}
        stud_psds1 = compute_psds_from_operating_points(stud_test_psds_buffer, test_tsv, test_dur,
                                                        save_dir=psds_folders[0], **psds1_kwargs)
        stud_psds2 = compute_psds_from_operating_points(stud_test_psds_buffer, test_tsv, test_dur,
                                                        save_dir=psds_folders[0], **psds2_kwargs)
        tch_psds1 = compute_psds_from_operating_points(tch_test_psds_buffer, test_tsv, test_dur,
                                                       save_dir=psds_folders[1], **psds1_kwargs)
        tch_psds2 = compute_psds_from_operating_points(tch_test_psds_buffer, test_tsv, test_dur,
                                                       save_dir=psds_folders[1], **psds2_kwargs)
    else:
        psds1_kwargs_true = {"ground_truth": test_tsv, "audio_durations": test_dur, "dtc_threshold": 0.7,
                             "gtc_threshold": 0.7, "alpha_ct": 0, "alpha_st": 1, "unit_of_time": 'hour', "max_efpr": 100.,
                             "num_jobs": train_cfg["num_workers"]}
        psds2_kwargs_true = {"ground_truth": test_tsv, "audio_durations": test_dur, "dtc_threshold": 0.1,
                             "gtc_threshold": 0.1, "cttc_threshold": 0.3, "alpha_ct": 0.5, "alpha_st": 1,
                             "unit_of_time": 'hour', "max_efpr": 100., "num_jobs": train_cfg["num_workers"]}
        stud_psds1, _, _ = intersection_based.psds(scores=stud_scores, **psds1_kwargs_true)
        stud_psds2, _, _ = intersection_based.psds(scores=stud_scores, **psds2_kwargs_true)
        tch_psds1, _, _ = intersection_based.psds(scores=tch_scores, **psds1_kwargs_true)
        tch_psds2, _, _ = intersection_based.psds(scores=tch_scores, **psds2_kwargs_true)
    stud_CB_f1, _, _, _ = log_sedeval_metrics(stud_test_f1_buffer, test_tsv, psds_folders[0])
    stud_IB_f1 = compute_per_intersection_macro_f1({"0.5": stud_test_f1_buffer}, test_tsv, test_dur)
    tch_CB_f1, _, _, _ = log_sedeval_metrics(tch_test_f1_buffer, test_tsv, psds_folders[1])
    tch_IB_f1 = compute_per_intersection_macro_f1({"0.5": tch_test_f1_buffer}, test_tsv, test_dur)
    return stud_psds1, stud_psds2, stud_CB_f1, stud_IB_f1, tch_psds1, tch_psds2, tch_CB_f1, tch_IB_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gpu_selection")
    parser.add_argument('-c', '--config', default="./configs/config_MDFDbest_cwmf_tpsds.yaml", type=str)
    parser.add_argument('-g', '--gpu', default=0, type=int)
    parser.add_argument('-m', '--multigpu', default=False, type=bool)
    parser.add_argument('-r', '--repeat', default=2, type=int)
    parser.add_argument('-s', '--start', default=1, type=int)
    args = parser.parse_args()
    for iteration in range(args.start, args.repeat):
        main(args, iteration)
