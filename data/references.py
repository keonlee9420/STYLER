import numpy as np
import os
from tqdm import tqdm
import glob
import tgt
from scipy.io.wavfile import read
import pyworld as pw
import torch
import audio as Audio
import utils
from text import _clean_text
import hparams as hp


def prepare_align(in_dir):
    for dirpath, dirnames, filenames in tqdm(os.walk(in_dir)):
        for file in filenames:
            if file.endswith(".txt"):
                path_in  = os.path.join(dirpath, file)
                with open(path_in, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    assert(len(lines) == 1)
                    text = lines[0]
                    text = _clean_text(text, hp.text_cleaners)

                path_out = os.path.join(dirpath, file)
                with open(path_out, 'w', encoding='utf-8') as f:
                    f.write(text)

def build_from_path(in_dir, out_dir):
    ref_dir_name = in_dir.split("/")[-1]
    basenames = []
    audio_paths = list(glob.glob(os.path.join(in_dir, '*.wav')))
    index = 1
    f0_max = energy_max = 0
    f0_min = energy_min = 1000000
    n_frames = 0
    filelist_list = list()
    for ref_path in glob.glob(os.path.join(in_dir, '*.wav')):
        basename = ref_path.split("/")[-1].replace(".wav","")
        text = utils.get_transcript(ref_path.replace(".wav", ".txt"))
        filelist_list.append("|".join([basename, text]))

        try:
            ret = process_utterance(in_dir, out_dir, basename)
            if ret is None:
                continue
            else:
                info, f_max, f_min, e_max, e_min, n = ret

            print("Done {}: {}".format(index, basename))
            basenames.append(basename)

            index = index + 1

            f0_max = max(f0_max, f_max)
            f0_min = min(f0_min, f_min)
            energy_max = max(energy_max, e_max)
            energy_min = min(energy_min, e_min)
            n_frames += n
        except:
            print("Can't process:", basename)

    strs = ['Total time: {} hours'.format(n_frames*hp.hop_length/hp.sampling_rate/3600),
            'Total frames: {}'.format(n_frames),
            'Min F0: {}'.format(f0_min),
            'Max F0: {}'.format(f0_max),
            'Min energy: {}'.format(energy_min),
            'Max energy: {}'.format(energy_max)]
    for s in strs:
        print(s)
    
    ### Write Filelist ###
    with open(os.path.join(out_dir, '{}.txt'.format(ref_dir_name)), 'w', encoding='utf-8') as f:
        print("Total saved filelist elements:", len(filelist_list))
        for row in filelist_list:
            f.write(str(row)+'\n')

    return basenames, audio_paths


def process_utterance(in_dir, out_dir, basename):
    wav_path = os.path.join(in_dir, '{}.wav'.format(basename))
    tg_path = os.path.join(out_dir, 'TextGrid', '{}.TextGrid'.format(basename))

    # Get alignments
    textgrid = tgt.io.read_textgrid(tg_path)
    phone, duration, start, end = utils.get_alignment(
        textgrid.get_tier_by_name('phones'))
    # '{A}{B}{$}{C}', $ represents silent phones
    text = '{' + '}{'.join(phone) + '}'
    text = text.replace('{$}', ' ')    # '{A}{B} {C}'
    text = text.replace('}{', ' ')     # '{A B} {C}'

    if start >= end:
        return None

    # Read and trim wav files
    _, wav = read(wav_path)
    wav = wav[int(hp.sampling_rate*start):int(hp.sampling_rate*end)].astype(np.float32)

    # Compute fundamental frequency
    f0, _ = pw.dio(wav.astype(np.float64), hp.sampling_rate,
                   frame_period=hp.hop_length/hp.sampling_rate*1000)
    f0 = f0[:sum(duration)]

    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy, _ = Audio.tools.get_mel_from_wav(
        torch.FloatTensor(wav))
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[
        :, :sum(duration)]
    energy = energy.numpy().astype(np.float32)[:sum(duration)]
    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None

    # Save alignment
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'alignment', ali_filename),
            duration, allow_pickle=False)

    # Save fundamental prequency
    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)

    # Save normalized fundamental prequency
    f0_norm = utils.f0_normalization(f0)
    np.save(os.path.join(out_dir, 'f0_norm', f0_filename), f0_norm, allow_pickle=False)

    # Save energy
    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'energy', energy_filename),
            energy, allow_pickle=False)

    # Save rescaled energy
    energy_0to1 = utils.energy_rescaling(energy)
    np.save(os.path.join(out_dir, 'energy_0to1', energy_filename), energy_0to1, allow_pickle=False)

    # Save spectrogram
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'mel', mel_filename),
            mel_spectrogram.T, allow_pickle=False)

    return '|'.join([basename, text]), max(f0), min([f for f in f0 if f != 0]), max(energy), min(energy), mel_spectrogram.shape[1]
