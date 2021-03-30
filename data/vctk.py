import numpy as np
import os
import random
from tqdm import tqdm
import re
import tgt
from scipy.io.wavfile import read
import pyworld as pw
import torch
import audio as Audio
from text import _clean_text
import hparams as hp
from pathlib import Path
from deepspeaker import embedding
import utils


def write_filelist(data, outdir, filename):
    with open(os.path.join(outdir, filename + ".txt"), 'w', encoding='utf-8') as f:
         for d in data:
            if d is None:
                continue
            f.write(d + '\n')
    f.close()

### prepare align ###
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

def get_unaligned_wavs(out_dir):
    unaligned=list()
    if os.path.isfile(os.path.join(out_dir, 'output_errors.txt')):
        with open(os.path.join(out_dir, 'output_errors.txt'), encoding='utf-8') as f:
            all_txt = f.read()
            all_txt = all_txt.split(":\nTraceback")
            unaligned += [t.split('\n')[-1] for t in all_txt if 'CB' in t]
    if os.path.isfile(os.path.join(out_dir, 'unaligned.txt')):
        with open(os.path.join(out_dir, 'unaligned.txt'), encoding='utf-8') as f:
            for line in f:
                unaligned.append(line.strip().split(' ')[0].split('\t')[0])
    return unaligned

### Creating dataset ###
def build_from_path(in_dir, out_dir):
    random.seed(9420)
    train = list()
    val = list()
    f0_max = energy_max = 0
    f0_min = energy_min = 1000000
    n_frames = 0
    max_text_len = 0
    max_mel_len = 0

    embedder = embedding.build_model(hp.speaker_embedder_dir)

    spkers = os.listdir(in_dir)
    spkers.sort()
    print("Total Speakers : {}".format(len(spkers)))

    unaligned_basenames = get_unaligned_wavs(out_dir)
    print("Total unaligned wavs: {}".format(len(unaligned_basenames)))
    dirty_basenames = list()
    error_basenames = list()

    # Calculate mean speaker embedding for each speaker
    if not (len(os.listdir(os.path.join(out_dir, 'spker_embed'))) == len(spkers)):
        print("Speaker embedding...")
        for spker in tqdm(spkers):
            spker_embedding = list()
            spker_dir = os.path.join(in_dir, spker)
            for dirpath, dirnames, filenames in os.walk(spker_dir):
                for f in filenames:
                    if f.endswith(".wav"):
                        if f.replace(".wav", "") in unaligned_basenames:
                            continue
                        subdir = Path(dirpath).relative_to(in_dir)
                        audio_path = os.path.join(dirpath, f)
                        spker_embedding.append(embedding.predict_embedding(embedder, audio_path))
            spker_embed = np.mean(spker_embedding, axis=0)

            # Save speaker embedding
            spker_embed_filename = '{}-spker_embed-{}.npy'.format(hp.dataset, spker)
            np.save(os.path.join(out_dir, 'spker_embed', spker_embed_filename), spker_embed, allow_pickle=False)

    if not os.path.exists(os.path.join(out_dir, "TextGrid")):
        raise FileNotFoundError("\"TextGird\" not found in {}".format(out_dir))

    print("Process utterances...")
    for spker in tqdm(spkers):
        spker_dir = os.path.join(in_dir, spker)
        file_paths = []
        for dirpath, dirnames, filenames in os.walk(spker_dir):
            for f in filenames:
                if f.endswith(".txt"):
                    if f.replace(".txt", "") in unaligned_basenames:
                        continue
                    subdir = Path(dirpath).relative_to(in_dir)
                    file_paths.append((subdir, f))

        random.shuffle(file_paths)
        for i, file_path in enumerate(file_paths):
            subdir = file_path[0]
            filename = file_path[1]
            basename = filename.replace(".txt", "")
            try:
                ret = process_utterance(in_dir, out_dir, subdir, basename)
            except:
                error_basenames.append(basename)
                continue
            if ret is None:
                dirty_basenames.append(basename)
                continue
            else:
                info, f_max, f_min, e_max, e_min, n = ret
                text = info.split('|')[-1]
                
            if i == 0: 
                val.append(info)
            else:
                train.append(info)
                
            f0_max = max(f0_max, f_max)
            f0_min = min(f0_min, f_min)
            energy_max = max(energy_max, e_max)
            energy_min = min(energy_min, e_min)
            n_frames += n
            max_text_len = max(max_text_len, len(text))
            max_mel_len = max(max_mel_len, n)

    print("Total dirty wavs: {}".format(len(dirty_basenames)))
    print("Total error wavs: {}".format(len(error_basenames)))

    ### Write Stats ###
    with open(os.path.join(out_dir, 'stat.txt'), 'w', encoding='utf-8') as f:
        strs = ['Total files: {}'.format(len(train)+len(val)),
                'Total time: {} hours'.format(n_frames*hp.hop_length/hp.sampling_rate/3600),
                'Total frames: {}'.format(n_frames),
                'Min F0: {}'.format(f0_min),
                'Max F0: {}'.format(f0_max),
                'Min energy: {}'.format(energy_min),
                'Max energy: {}'.format(energy_max),
                'Max text len: {}'.format(max_text_len),
                'Max mel len: {}'.format(max_mel_len),
                'Total unaligned wavs: {}'.format(len(unaligned_basenames)),
                'Total dirty wavs: {}'.format(len(dirty_basenames)),
                'Total error wavs: {}'.format(len(error_basenames))]
        for s in strs:
            print(s)
            f.write(s+'\n')

    # write filelists
    write_filelist(dirty_basenames, out_dir, "dirty")
    write_filelist(error_basenames, out_dir, "error")

    return [r for r in train if r is not None], [r for r in val if r is not None]

def process_utterance(in_dir, out_dir, dirname, basename):
    wav_path = os.path.join(in_dir , dirname   , '{}.wav'.format(basename))
    tg_path  = os.path.join(out_dir, 'TextGrid', dirname, '{}.TextGrid'.format(basename))

    if not os.path.exists(tg_path):
        return None
        
    # Get alignments
    textgrid = tgt.io.read_textgrid(tg_path)
    phone, duration, start, end = utils.get_alignment(textgrid.get_tier_by_name('phones'))
    text = '{'+ '}{'.join(phone) + '}' # '{A}{B}{$}{C}', $ represents silent phones
    text = text.replace('{$}', ' ')    # '{A}{B} {C}'
    text = text.replace('}{', ' ')     # '{A B} {C}'
    
    if start >= end:
        return None

    # Read and trim wav files
    sr, wav = read(wav_path)
    wav = wav[int(hp.sampling_rate*start):int(hp.sampling_rate*end)].astype(np.float32)

    # Compute fundamental frequency
    f0, _ = pw.dio(wav.astype(np.float64), hp.sampling_rate, frame_period=hp.hop_length/hp.sampling_rate*1000)
    f0 = f0[:sum(duration)]

    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy, _ = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav))
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[:, :sum(duration)]
    energy = energy.numpy().astype(np.float32)[:sum(duration)]
    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None

    # Save alignment
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'alignment', ali_filename), duration, allow_pickle=False)

    # Save fundamental prequency
    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)

    # Save normalized fundamental prequency
    f0_norm = utils.f0_normalization(f0)
    np.save(os.path.join(out_dir, 'f0_norm', f0_filename), f0_norm, allow_pickle=False)

    # Save energy
    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'energy', energy_filename), energy, allow_pickle=False)

    # Save rescaled energy
    energy_0to1 = utils.energy_rescaling(energy)
    np.save(os.path.join(out_dir, 'energy_0to1', energy_filename), energy_0to1, allow_pickle=False)

    # Save spectrogram
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'mel_clean', mel_filename), mel_spectrogram.T, allow_pickle=False)
    return '|'.join([basename, text]), max(f0), min([f for f in f0 if f != 0]), max(energy), min(energy), mel_spectrogram.shape[1]