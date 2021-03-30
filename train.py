import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import argparse
import os
import time

from styler import STYLER
from loss import STYLERLoss, DomainAdversarialTrainingLoss
from dataset import Dataset
from optimizer import ScheduledOptim
from evaluate import evaluate
import hparams as hp
import utils
import audio as Audio


def main(args):
    torch.manual_seed(0)

    # Get device
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')

    # Get dataset
    dataset = Dataset("train.txt")
    loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=True,
                        collate_fn=dataset.collate_fn, drop_last=True, num_workers=0)

    # Define model
    model = nn.DataParallel(STYLER()).to(device)
    print("Model Has Been Defined")
    
    # Parameters
    num_param = utils.get_param_num(model)
    text_encoder = utils.get_param_num(model.module.style_modeling.style_encoder.text_encoder)
    audio_encoder = utils.get_param_num(model.module.style_modeling.style_encoder.audio_encoder)
    predictors = utils.get_param_num(model.module.style_modeling.duration_predictor)\
         + utils.get_param_num(model.module.style_modeling.pitch_predictor)\
              + utils.get_param_num(model.module.style_modeling.energy_predictor)
    decoder = utils.get_param_num(model.module.decoder)
    print('Number of Model Parameters          :', num_param)
    print('Number of Text Encoder Parameters   :', text_encoder)
    print('Number of Audio Encoder Parameters  :', audio_encoder)
    print('Number of Predictor Parameters      :', predictors)
    print('Number of Decoder Parameters        :', decoder)

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), betas=hp.betas, eps=hp.eps, weight_decay=hp.weight_decay)
    scheduled_optim = ScheduledOptim(
        optimizer, hp.decoder_hidden, hp.n_warm_up_step, args.restore_step)
    Loss = STYLERLoss().to(device)
    DATLoss = DomainAdversarialTrainingLoss().to(device)
    print("Optimizer and Loss Function Defined.")

    # Load checkpoint if exists
    checkpoint_path = os.path.join(hp.checkpoint_path())
    try:
        checkpoint = torch.load(os.path.join(
            checkpoint_path, 'checkpoint_{}.pth.tar'.format(args.restore_step)))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step {}---\n".format(args.restore_step))
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # Load vocoder
    vocoder = utils.get_vocoder()

    # Init logger
    log_path = hp.log_path()
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(os.path.join(log_path, 'train'))
        os.makedirs(os.path.join(log_path, 'validation'))
    train_logger = SummaryWriter(os.path.join(log_path, 'train'))
    val_logger = SummaryWriter(os.path.join(log_path, 'validation'))

    # Init synthesis directory
    synth_path = hp.synth_path()
    if not os.path.exists(synth_path):
        os.makedirs(synth_path)

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    model = model.train()
    for epoch in range(hp.epochs):
        # Get Training Loader
        total_step = hp.epochs * len(loader) * hp.batch_size

        for i, batchs in enumerate(loader):
            for j, data_of_batch in enumerate(batchs):
                start_time = time.perf_counter()

                current_step = i*hp.batch_size + j + args.restore_step + \
                    epoch*len(loader)*hp.batch_size + 1

                # Get Data
                text = torch.from_numpy(
                    data_of_batch["text"]).long().to(device)
                mel_target = torch.from_numpy(
                    data_of_batch["mel_target"]).float().to(device)
                mel_aug = torch.from_numpy(
                    data_of_batch["mel_aug"]).float().to(device)
                D = torch.from_numpy(data_of_batch["D"]).long().to(device)
                log_D = torch.from_numpy(
                    data_of_batch["log_D"]).float().to(device)
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

                # Forward
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

                # Total loss
                total_loss = mel_loss + mel_postnet_loss + mel_noisy_loss + mel_postnet_noisy_loss + d_loss + f_loss + e_loss\
                    + hp.dat_weight*(classifier_loss_a + classifier_loss_a_dat)

                # Logger
                t_l = total_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                m_n_l = mel_noisy_loss.item()
                m_p_n_l = mel_postnet_noisy_loss.item()
                d_l = d_loss.item()
                f_l = f_loss.item()
                e_l = e_loss.item()
                cl_a = classifier_loss_a.item()
                cl_a_dat = classifier_loss_a_dat.item()

                # Backward
                total_loss = total_loss / hp.acc_steps
                total_loss.backward()
                if current_step % hp.acc_steps != 0:
                    continue

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), hp.grad_clip_thresh)

                # Update weights
                scheduled_optim.step_and_update_lr()
                scheduled_optim.zero_grad()

                # Print
                if current_step == 1 or current_step % hp.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch+1, hp.epochs, current_step, total_step)
                    str2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, F0 Loss: {:.4f}, Energy Loss: {:.4f};".format(
                        t_l, m_l, m_p_l, d_l, f_l, e_l)
                    str3 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                        (Now-Start), (total_step-current_step)*np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    print(str3)

                    train_logger.add_scalar(
                        'Loss/total_loss', t_l, current_step)
                    train_logger.add_scalar('Loss/mel_loss', m_l, current_step)
                    train_logger.add_scalar(
                        'Loss/mel_postnet_loss', m_p_l, current_step)
                    train_logger.add_scalar('Loss/mel_noisy_loss', m_n_l, current_step)
                    train_logger.add_scalar(
                        'Loss/mel_postnet_noisy_loss', m_p_n_l, current_step)
                    train_logger.add_scalar(
                        'Loss/duration_loss', d_l, current_step)
                    train_logger.add_scalar('Loss/F0_loss', f_l, current_step)
                    train_logger.add_scalar(
                        'Loss/energy_loss', e_l, current_step)
                    train_logger.add_scalar(
                        'Loss/dat_clean_loss', cl_a, current_step)
                    train_logger.add_scalar(
                        'Loss/dat_noisy_loss', cl_a_dat, current_step)

                if current_step % hp.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(current_step)))
                    print("save model at step {} ...".format(current_step))

                if current_step == 1 or current_step % hp.synth_step == 0:
                    length = mel_len[0].item()
                    mel_target_torch = mel_target[0, :length].detach(
                    ).unsqueeze(0).transpose(1, 2)
                    mel_aug_torch = mel_aug[0, :length].detach(
                    ).unsqueeze(0).transpose(1, 2)
                    mel_target = mel_target[0, :length].detach(
                    ).cpu().transpose(0, 1)
                    mel_aug = mel_aug[0, :length].detach(
                    ).cpu().transpose(0, 1)
                    mel_torch = mel_output[0, :length].detach(
                    ).unsqueeze(0).transpose(1, 2)
                    mel_noisy_torch = mel_output_noisy[0, :length].detach(
                    ).unsqueeze(0).transpose(1, 2)
                    mel = mel_output[0, :length].detach().cpu().transpose(0, 1)
                    mel_noisy = mel_output_noisy[0, :length].detach().cpu().transpose(0, 1)
                    mel_postnet_torch = mel_postnet_output[0, :length].detach(
                    ).unsqueeze(0).transpose(1, 2)
                    mel_postnet_noisy_torch = mel_postnet_output_noisy[0, :length].detach(
                    ).unsqueeze(0).transpose(1, 2)
                    mel_postnet = mel_postnet_output[0, :length].detach(
                    ).cpu().transpose(0, 1)
                    mel_postnet_noisy = mel_postnet_output_noisy[0, :length].detach(
                    ).cpu().transpose(0, 1)
                    # Audio.tools.inv_mel_spec(mel, os.path.join(
                    #     synth_path, "step_{}_{}_griffin_lim.wav".format(current_step, "c")))
                    # Audio.tools.inv_mel_spec(mel_postnet, os.path.join(
                    #     synth_path, "step_{}_{}_postnet_griffin_lim.wav".format(current_step, "c")))
                    # Audio.tools.inv_mel_spec(mel_noisy, os.path.join(
                    #     synth_path, "step_{}_{}_griffin_lim.wav".format(current_step, "n")))
                    # Audio.tools.inv_mel_spec(mel_postnet_noisy, os.path.join(
                    #     synth_path, "step_{}_{}_postnet_griffin_lim.wav".format(current_step, "n")))

                    wav_mel = utils.vocoder_infer(mel_torch, vocoder, os.path.join(
                        hp.synth_path(), 'step_{}_{}_{}.wav'.format(current_step, "c", hp.vocoder)))
                    wav_mel_postnet = utils.vocoder_infer(mel_postnet_torch, vocoder, os.path.join(
                        hp.synth_path(), 'step_{}_{}_postnet_{}.wav'.format(current_step, "c", hp.vocoder)))
                    wav_ground_truth = utils.vocoder_infer(mel_target_torch, vocoder, os.path.join(
                        hp.synth_path(), 'step_{}_{}_ground-truth_{}.wav'.format(current_step, "c", hp.vocoder)))
                    wav_mel_noisy = utils.vocoder_infer(mel_noisy_torch, vocoder, os.path.join(
                        hp.synth_path(), 'step_{}_{}_{}.wav'.format(current_step, "n", hp.vocoder)))
                    wav_mel_postnet_noisy = utils.vocoder_infer(mel_postnet_noisy_torch, vocoder, os.path.join(
                        hp.synth_path(), 'step_{}_{}_postnet_{}.wav'.format(current_step, "n", hp.vocoder)))
                    wav_aug = utils.vocoder_infer(mel_aug_torch, vocoder, os.path.join(
                        hp.synth_path(), 'step_{}_{}_ground-truth_{}.wav'.format(current_step, "n", hp.vocoder)))

                    # Model duration prediction
                    log_duration_output = log_duration_output[0, :src_len[0].item()].detach().cpu() # [seg_len]
                    log_duration_output = torch.clamp(torch.round(torch.exp(log_duration_output)-hp.log_offset), min=0).int()
                    model_duration = utils.get_alignment_2D(log_duration_output).T # [seg_len, mel_len]
                    model_duration = utils.plot_alignment([model_duration])

                    # Model mel prediction
                    f0 = f0[0, :length].detach().cpu().numpy()
                    energy = energy[0, :length].detach().cpu().numpy()
                    f0_output = f0_output[0, :length].detach().cpu().numpy()
                    energy_output = energy_output[0,
                                                  :length].detach().cpu().numpy()
                    mel_predicted = utils.plot_data([(mel_postnet.numpy(), f0_output, energy_output), (mel_target.numpy(), f0, energy)],
                                    ['Synthetized Spectrogram Clean', 'Ground-Truth Spectrogram'], filename=os.path.join(synth_path, 'step_{}_{}.png'.format(current_step, "c")))
                    mel_noisy_predicted = utils.plot_data([(mel_postnet_noisy.numpy(), f0_output, energy_output), (mel_aug.numpy(), f0, energy)],
                                    ['Synthetized Spectrogram Noisy', 'Aug Spectrogram'], filename=os.path.join(synth_path, 'step_{}_{}.png'.format(current_step, "n")))

                    # Normalize audio for tensorboard logger. See https://github.com/lanpa/tensorboardX/issues/511#issuecomment-537600045
                    wav_ground_truth = wav_ground_truth/max(wav_ground_truth)
                    wav_mel = wav_mel/max(wav_mel)
                    wav_mel_postnet = wav_mel_postnet/max(wav_mel_postnet)
                    wav_aug = wav_aug/max(wav_aug)
                    wav_mel_noisy = wav_mel_noisy/max(wav_mel_noisy)
                    wav_mel_postnet_noisy = wav_mel_postnet_noisy/max(wav_mel_postnet_noisy)

                    train_logger.add_image(
                        "model_duration",
                        model_duration,
                        current_step, dataformats='HWC')
                    train_logger.add_image(
                        "mel_predicted/Clean",
                        mel_predicted,
                        current_step, dataformats='HWC')
                    train_logger.add_image(
                        "mel_predicted/Noisy",
                        mel_noisy_predicted,
                        current_step, dataformats='HWC')
                    train_logger.add_audio(
                        "Clean/wav_ground_truth",
                        wav_ground_truth,
                        current_step, sample_rate=hp.sampling_rate)
                    train_logger.add_audio(
                        "Clean/wav_mel",
                        wav_mel,
                        current_step, sample_rate=hp.sampling_rate)
                    train_logger.add_audio(
                        "Clean/wav_mel_postnet",
                        wav_mel_postnet,
                        current_step, sample_rate=hp.sampling_rate)
                    train_logger.add_audio(
                        "Noisy/wav_aug",
                        wav_aug,
                        current_step, sample_rate=hp.sampling_rate)
                    train_logger.add_audio(
                        "Noisy/wav_mel_noisy",
                        wav_mel_noisy,
                        current_step, sample_rate=hp.sampling_rate)
                    train_logger.add_audio(
                        "Noisy/wav_mel_postnet_noisy",
                        wav_mel_postnet_noisy,
                        current_step, sample_rate=hp.sampling_rate)

                if current_step == 1 or current_step % hp.eval_step == 0:
                    model.eval()
                    with torch.no_grad():
                        d_l, f_l, e_l, cl_a, cl_a_dat, m_l, m_p_l, m_n_l, m_p_n_l = evaluate(
                            model, current_step)
                        t_l = d_l + f_l + e_l + m_l + m_p_l + m_n_l + m_p_n_l\
                            + hp.dat_weight*(cl_a + cl_a_dat)

                        val_logger.add_scalar(
                            'Loss/total_loss', t_l, current_step)
                        val_logger.add_scalar(
                            'Loss/mel_loss', m_l, current_step)
                        val_logger.add_scalar(
                            'Loss/mel_postnet_loss', m_p_l, current_step)
                        val_logger.add_scalar(
                            'Loss/mel_noisy_loss', m_n_l, current_step)
                        val_logger.add_scalar(
                            'Loss/mel_postnet_noisy_loss', m_p_n_l, current_step)
                        val_logger.add_scalar(
                            'Loss/duration_loss', d_l, current_step)
                        val_logger.add_scalar(
                            'Loss/F0_loss', f_l, current_step)
                        val_logger.add_scalar(
                            'Loss/energy_loss', e_l, current_step)
                        val_logger.add_scalar(
                            'Loss/dat_clean_loss', cl_a, current_step)
                        val_logger.add_scalar(
                            'Loss/dat_noisy_loss', cl_a_dat, current_step)

                    model.train()

                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--version', type=str, default="default")
    parser.add_argument('--batch_size', type=int, default=hp.batch_size)
    args = parser.parse_args()

    # Set Batch Size
    hp.batch_size = args.batch_size

    # Version Control
    hp.version = args.version + "_batch{}".format(hp.batch_size)

    main(args)
