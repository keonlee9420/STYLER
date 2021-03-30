import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os
import tgt
from scipy.io.wavfile import read
import pyworld as pw
from pysptk import sptk

import hparams
import audio as Audio
from utils import pad_1D, pad_2D, process_meta, get_alignment
from text import text_to_sequence, sequence_to_text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_preprocessed_wav(wav_path, tg_path):
    # Get alignments
    textgrid = tgt.io.read_textgrid(tg_path)
    _, duration, start, end = get_alignment(
        textgrid.get_tier_by_name('phones'))

    # Read and trim wav files
    sr, wav = read(wav_path)
    wav = wav[int(hparams.sampling_rate*start):int(hparams.sampling_rate*end)].astype(np.float32)
    return wav, sr, duration


def get_f0(wav, duration=None):
    f0, _ = pw.dio(wav.astype(np.float64), hparams.sampling_rate,
                   frame_period=hparams.hop_length/hparams.sampling_rate*1000)
    if duration is not None:
        f0 = f0[:sum(duration)]
    return f0


def get_f0_noisy(wav, duration=None):
    f0 = sptk.rapt(wav.astype(np.float32)*hparams.max_wav_value, hparams.sampling_rate, hparams.encoder_hidden, min=hparams.f0_min, max=hparams.f0_max, otype=2) # log f0
    if duration is not None:
        f0 = f0[:sum(duration)]
    f0 = np.exp(f0)
    return f0


def get_mel_and_energy(wav, duration, norm=True):
    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy, clipt = Audio.tools.get_mel_from_wav(
            torch.FloatTensor(wav), norm=norm)
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[
        :, :sum(duration)]
    energy = energy.numpy().astype(np.float32)[:sum(duration)]
    return mel_spectrogram, energy, clipt


def get_processed_data_from_wav(wav_path, tg_path, noisy_input):
    # Get wav and duration
    wav, _, duration = get_preprocessed_wav(wav_path, tg_path)

    # Compute fundamental frequency
    if noisy_input:
        f0 = get_f0_noisy(wav, duration)
    else:
        f0 = get_f0(wav, duration)

    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy, _ = get_mel_and_energy(wav, duration)

    return f0, energy, mel_spectrogram.T


class Dataset(Dataset):
    def __init__(self, filename="train.txt", sort=True, speaker_lookup_table=None):
        self.basename, self.text = process_meta(
            os.path.join(hparams.preprocessed_path, filename))
        self.sort = sort
        self.speaker_lookup_table = speaker_lookup_table

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker_embed_path = os.path.join(
            hparams.preprocessed_path, "spker_embed", "{}-spker_embed-{}.npy".format(hparams.dataset, str(basename.split('_')[0])))
        speaker_embed = np.load(speaker_embed_path)
        phone = np.array(text_to_sequence(self.text[idx], []))
        mel_target_path = os.path.join(
            hparams.preprocessed_path, "mel_clean", "{}-mel-{}.npy".format(hparams.dataset, basename))
        mel_target = np.load(mel_target_path)
        mel_aug_path = os.path.join(
            hparams.preprocessed_path, "mel_aug", "{}-mel-{}.npy".format(hparams.dataset, basename))
        mel_aug = np.load(mel_aug_path)
        D_path = os.path.join(
            hparams.preprocessed_path, "alignment", "{}-ali-{}.npy".format(hparams.dataset, basename))
        D = np.load(D_path)
        f0_path = os.path.join(
            hparams.preprocessed_path, "f0", "{}-f0-{}.npy".format(hparams.dataset, basename))
        f0 = np.load(f0_path)
        f0_norm_path = os.path.join(
            hparams.preprocessed_path, "f0_norm", "{}-f0-{}.npy".format(hparams.dataset, basename))
        f0_norm = np.load(f0_norm_path)
        f0_norm_aug_path = os.path.join(
            hparams.preprocessed_path, "f0_norm_aug", "{}-f0-{}.npy".format(hparams.dataset, basename))
        f0_norm_aug = np.load(f0_norm_aug_path)
        energy_path = os.path.join(
            hparams.preprocessed_path, "energy", "{}-energy-{}.npy".format(hparams.dataset, basename))
        energy = np.load(energy_path)
        energy_input_path = os.path.join(
            hparams.preprocessed_path, "energy_0to1", "{}-energy-{}.npy".format(hparams.dataset, basename))
        energy_input = np.load(energy_input_path)
        energy_input_aug_path = os.path.join(
            hparams.preprocessed_path, "energy_0to1_aug", "{}-energy-{}.npy".format(hparams.dataset, basename))
        energy_input_aug = np.load(energy_input_aug_path)

        sample = {"id": basename,
                  "text": phone,
                  "mel_target": mel_target,
                  "mel_aug": mel_aug,
                  "D": D,
                  "f0": f0,
                  "f0_norm": f0_norm,
                  "f0_norm_aug": f0_norm_aug,
                  "energy": energy,
                  "energy_input": energy_input,
                  "energy_input_aug": energy_input_aug,
                  "speaker_embed": speaker_embed}

        return sample

    def reprocess(self, batch, cut_list):
        ids = [batch[ind]["id"] for ind in cut_list]
        texts = [batch[ind]["text"] for ind in cut_list]
        mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
        mel_augs = [batch[ind]["mel_aug"] for ind in cut_list]
        Ds = [batch[ind]["D"] for ind in cut_list]
        f0s = [batch[ind]["f0"] for ind in cut_list]
        f0_norms = [batch[ind]["f0_norm"] for ind in cut_list]
        f0_norm_augs = [batch[ind]["f0_norm_aug"] for ind in cut_list]
        energies = [batch[ind]["energy"] for ind in cut_list]
        energy_inputs = [batch[ind]["energy_input"] for ind in cut_list]
        energy_input_augs = [batch[ind]["energy_input_aug"] for ind in cut_list]
        speaker_embed = [batch[ind]["speaker_embed"] for ind in cut_list]
        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                print(text, text.shape, D, D.shape, id_)
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])

        texts = pad_1D(texts)
        Ds = pad_1D(Ds)
        mel_targets = pad_2D(mel_targets)
        mel_augs = pad_2D(mel_augs)
        f0s = pad_1D(f0s)
        f0_norms = pad_1D(f0_norms)
        f0_norm_augs = pad_1D(f0_norm_augs)
        energies = pad_1D(energies)
        energy_inputs = pad_1D(energy_inputs)
        energy_input_augs = pad_1D(energy_input_augs)
        log_Ds = np.log(Ds + hparams.log_offset)
        speaker_embeds = np.concatenate(speaker_embed, axis=0)

        out = {"id": ids,
               "text": texts,
               "mel_target": mel_targets,
               "mel_aug": mel_augs,
               "D": Ds,
               "log_D": log_Ds,
               "f0": f0s,
               "f0_norm": f0_norms,
               "f0_norm_aug": f0_norm_augs,
               "energy": energies,
               "energy_input": energy_inputs,
               "energy_input_aug": energy_input_augs,
               "speaker_embed": speaker_embeds,
               "src_len": length_text,
               "mel_len": length_mel}

        return out

    def collate_fn(self, batch):
        len_arr = np.array([d["text"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = int(math.sqrt(batchsize))

        cut_list = list()
        for i in range(real_batchsize):
            if self.sort:
                cut_list.append(
                    index_arr[i*real_batchsize:(i+1)*real_batchsize])
            else:
                cut_list.append(
                    np.arange(i*real_batchsize, (i+1)*real_batchsize))

        output = list()
        for i in range(real_batchsize):
            output.append(self.reprocess(batch, cut_list[i]))

        return output


if __name__ == "__main__":
    # Test
    dataset = Dataset('val.txt')
    training_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn,
                                 drop_last=True, num_workers=0)
    total_step = hparams.epochs * len(training_loader) * hparams.batch_size

    cnt = 0
    for i, batchs in enumerate(training_loader):
        for j, data_of_batch in enumerate(batchs):
            mel_target = torch.from_numpy(
                data_of_batch["mel_target"]).float().to(device)
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            if mel_target.shape[1] == D.sum().item():
                cnt += 1

    print(cnt, len(dataset))
