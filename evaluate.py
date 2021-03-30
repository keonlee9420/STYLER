import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import os

from styler import STYLER
from loss import STYLERLoss, DomainAdversarialTrainingLoss
from dataset import Dataset
import hparams as hp
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(num):
    checkpoint_path = os.path.join(
        hp.checkpoint_path(), "checkpoint_{}.pth.tar".format(num))
    model = nn.DataParallel(STYLER())
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.requires_grad = False
    model.eval()
    return model


def evaluate(model, step, vocoder=None):
    torch.manual_seed(0)

    # Get dataset
    dataset = Dataset("val.txt", sort=False)
    loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=False,
                        collate_fn=dataset.collate_fn, drop_last=False, num_workers=0, )

    # Get loss function
    Loss = STYLERLoss().to(device)
    DATLoss = DomainAdversarialTrainingLoss().to(device)

    # Evaluation
    d_l = []
    f_l = []
    e_l = []
    cl_a = []
    cl_a_dat = []
    mel_l = []
    mel_p_l = []
    mel_n_l = []
    mel_p_n_l = []

    current_step = 0
    idx = 0
    for i, batchs in enumerate(loader):
        for j, data_of_batch in enumerate(batchs):
            # Get Data
            id_ = data_of_batch["id"]
            text = torch.from_numpy(data_of_batch["text"]).long().to(device)
            mel_target = torch.from_numpy(
                data_of_batch["mel_target"]).float().to(device)
            mel_aug = torch.from_numpy(
                data_of_batch["mel_aug"]).float().to(device)
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            log_D = torch.from_numpy(data_of_batch["log_D"]).int().to(device)
            f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
            f0_norm = torch.from_numpy(data_of_batch["f0_norm"]).float().to(device)
            f0_norm_aug = torch.from_numpy(data_of_batch["f0_norm_aug"]).float().to(device)
            energy = torch.from_numpy(
                data_of_batch["energy"]).float().to(device)
            energy_input = torch.from_numpy(
                data_of_batch["energy_input"]).float().to(device)
            energy_input_aug = torch.from_numpy(
                data_of_batch["energy_input_aug"]).float().to(device)
            speaker_embed = torch.from_numpy(
                data_of_batch["speaker_embed"]).float().to(device)
            src_len = torch.from_numpy(
                data_of_batch["src_len"]).long().to(device)
            mel_len = torch.from_numpy(
                data_of_batch["mel_len"]).long().to(device)
            max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
            max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)

            with torch.no_grad():
                ## Forward
                mel_outputs, mel_postnet_outputs, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _, aug_posteriors = model(
                    text, mel_target, mel_aug, f0_norm, energy_input, src_len, mel_len, D, f0, energy, max_src_len, max_mel_len, speaker_embed=speaker_embed)

                # Cal Loss Clean
                mel_output, mel_postnet_output = mel_outputs[0], mel_postnet_outputs[0]
                mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss, classifier_loss_a = Loss(
                    log_duration_output, log_D, f0_output, f0, energy_output, energy, mel_output, mel_postnet_output, mel_target, ~src_mask, ~mel_mask, src_len, mel_len,\
                        aug_posteriors, torch.zeros(mel_target.size(0)).long().to(device))

                # Cal Loss Noisy
                mel_output_noisy, mel_postnet_output_noisy = mel_outputs[1], mel_postnet_outputs[1]
                mel_noisy_loss, mel_postnet_noisy_loss = Loss.cal_mel_loss(mel_output_noisy, mel_postnet_output_noisy, mel_aug, ~mel_mask)

                # Forward DAT
                enc_cat = model.module.style_modeling.style_encoder.encoder_input_cat(mel_aug, f0_norm_aug, energy_input_aug, mel_aug)
                duration_encoding, pitch_encoding, energy_encoding, _ = model.module.style_modeling.style_encoder.audio_encoder(enc_cat, mel_len, src_len, mask=None)
                aug_posterior_d = model.module.style_modeling.augmentation_classifier_d(duration_encoding)
                aug_posterior_p = model.module.style_modeling.augmentation_classifier_p(pitch_encoding)
                aug_posterior_e = model.module.style_modeling.augmentation_classifier_e(energy_encoding)
                
                # Cal Loss DAT
                classifier_loss_a_dat = DATLoss((aug_posterior_d, aug_posterior_p, aug_posterior_e), torch.ones(mel_target.size(0)).long().to(device))

                d_l.append(d_loss.item())
                f_l.append(f_loss.item())
                e_l.append(e_loss.item())
                cl_a.append(classifier_loss_a.item())
                cl_a_dat.append(classifier_loss_a_dat.item())
                mel_l.append(mel_loss.item())
                mel_p_l.append(mel_postnet_loss.item())
                mel_n_l.append(mel_noisy_loss.item())
                mel_p_n_l.append(mel_postnet_noisy_loss.item())

            current_step += 1

    d_l = sum(d_l) / len(d_l)
    f_l = sum(f_l) / len(f_l)
    e_l = sum(e_l) / len(e_l)
    cl_a = sum(cl_a) / len(cl_a)
    cl_a_dat = sum(cl_a_dat) / len(cl_a_dat)
    mel_l = sum(mel_l) / len(mel_l)
    mel_p_l = sum(mel_p_l) / len(mel_p_l)
    mel_n_l = sum(mel_n_l) / len(mel_n_l)
    mel_p_n_l = sum(mel_p_n_l) / len(mel_p_n_l)

    str1 = "STYLER Step {},".format(step)
    str2 = "Duration Loss: {}".format(d_l)
    str3 = "F0 Loss: {}".format(f_l)
    str4 = "Energy Loss: {}".format(e_l)
    str5 = "Mel Loss: {}".format(mel_l)
    str6 = "Mel Postnet Loss: {}".format(mel_p_l)

    print("\n" + str1)
    print(str2)
    print(str3)
    print(str4)
    print(str5)
    print(str6)

    return d_l, f_l, e_l, cl_a, cl_a_dat, mel_l, mel_p_l, mel_n_l, mel_p_n_l