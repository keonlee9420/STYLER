import numpy as np
import os
import random
from tqdm import tqdm
from scipy.io import wavfile
from scipy.io.wavfile import read
import torch
import audio as Audio
import hparams as hp
from dataset import get_preprocessed_wav, get_f0, get_f0_noisy, get_mel_and_energy
import utils
import glob
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import math
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

random.seed(9420)

# Mixer refers to https://github.com/microsoft/MS-SNSD
# Function to read audio
def audioread(path, tg_path=None, norm=True, start=0, stop=None):
    duration = None
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        if tg_path:
            x, sr, duration = get_preprocessed_wav(path, tg_path)
        else:
            x, sr = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')

    if len(x.shape) == 1:  # mono
        if norm:
            rms = (x ** 2).mean() ** 0.5
            scalar = 10 ** (-25 / 20) / (rms)
            x = x * scalar
    else:  # multi-channel
        x = x.T
        x = x.sum(axis=0)/x.shape[0]
        if norm:
            rms = (x ** 2).mean() ** 0.5
            scalar = 10 ** (-25 / 20) / (rms)
            x = x * scalar
    return x, sr, duration

# Funtion to write audio    
def audiowrite(data, fs, destpath, norm=False):
    if norm:
        eps = 1e-6
        rms = (data ** 2).mean() ** 0.5
        scalar = 10 ** (-25 / 10) / (rms+eps)
        data = data * scalar
        if max(abs(data))>=1:
            data = data/max(abs(data), eps)
    
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)
    
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    
    sf.write(destpath, data, fs)
    return

# Function to mix clean speech and noise at various SNR levels
def snr_mixer(clean, noise, snr):
    # Normalizing to -25 dB FS
    rmsclean = (clean**2).mean()**0.5
    scalarclean = 10 ** (-25 / 20) / rmsclean
    clean = clean * scalarclean
    rmsclean = (clean**2).mean()**0.5

    rmsnoise = (noise**2).mean()**0.5
    scalarnoise = 10 ** (-25 / 20) /rmsnoise
    noise = noise * scalarnoise
    rmsnoise = (noise**2).mean()**0.5

    # Set the noise level for a given SNR
    noisescalar = np.sqrt(rmsclean / (10**(snr/20)) / rmsnoise)
    noisenewlevel = noise * noisescalar
    noisyspeech = clean + noisenewlevel
    return clean, noisenewlevel, noisyspeech


def get_unaligned_wavs(out_dir):
    unaligned=list()
    if os.path.isfile(os.path.join(out_dir, 'output_errors.txt')):
        with open(os.path.join(out_dir, 'output_errors.txt'), encoding='utf-8') as f:
            all_txt = f.read()
            all_txt = all_txt.split(":\nTraceback")
            unaligned += [t.split('\n')[-1] for t in all_txt if t.split('\n')[-1] != '']
    if os.path.isfile(os.path.join(out_dir, 'unaligned.txt')):
        with open(os.path.join(out_dir, 'unaligned.txt'), encoding='utf-8') as f:
            for line in f:
                unaligned.append(line.strip().split(' ')[0].split('\t')[0])
    return unaligned


def basenames_and_speakers(filelist_path):
    basenames = list()
    speakers = set()
    with open(filelist_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            basename = parts[0]
            basenames.append(basename)
            speaker_id = str(basename.split('_')[0])
            speakers.add(speaker_id)
    return basenames, speakers


def build_from_path(in_dir, out_dir,
                noise_dir=hp.noise_dir, 
                snr_lower=5, 
                snr_upper=25,
                silence_length=0.2,
                save_aux_max=10,
                noise_speaker_rate=0.5):
    tg_dir = os.path.join(out_dir, "TextGrid")
    spkers = os.listdir(in_dir)
    print("Total Speakers : {}".format(len(spkers)))

    unaligned_basenames = get_unaligned_wavs(out_dir)
    print("Total unaligned wavs: {}".format(len(unaligned_basenames)))

    cleanbasenames_train, speakers_train = basenames_and_speakers(os.path.join(out_dir, "train.txt"))
    cleanbasenames_val, speakers_val = basenames_and_speakers(os.path.join(out_dir, "val.txt"))
    print("Total train set : {}".format(len(cleanbasenames_train)))
    print("Total train speakers : {}".format(len(speakers_train)))

    print("Total val set : {}".format(len(cleanbasenames_val)))
    print("Total val speakers : {}".format(len(speakers_val)))

    noisefilenames = glob.glob(os.path.join(noise_dir, '*.wav'))
    print("Number of total noise files:", len(noisefilenames))

    # Shuffle and divide the noise data
    random.shuffle(noisefilenames)
    train_divider = 27900 # defines size of train and aug set
    val_divider = 100
    noisefilenames_train = noisefilenames[:train_divider]
    noisefilenames_val = noisefilenames[train_divider:]
    assert (train_divider+val_divider) <= len(noisefilenames), "Noise divider out of range"
    
    print("Total noise for train:", len(noisefilenames_train))
    print("Total noise for val:", len(noisefilenames_val))

    def noise_path_and_name(noisefilenames, idx):
        noise_path = noisefilenames[idx % len(noisefilenames)]
        noise_name = noise_path.split('/')[-1].replace(".wav","")
        return noise_path, noise_name

    def mixer(clean, noisefilenames, idx):
        noise_path, noise_name = noise_path_and_name(noisefilenames, idx)
        noise, _, _ = audioread(noise_path)

        if len(noise)>=len(clean):
            noise = noise[0:len(clean)]
        else:
            while len(noise)<=len(clean):
                noise_path_aux = noisefilenames[random.randint(0, len(noisefilenames)-1)]
                if noise_path_aux == noise_path: continue
                newnoise, sr_newnoise, _ = audioread(noise_path_aux)
                noiseconcat = np.append(noise, np.zeros(int(sr_newnoise*silence_length)))
                noise = np.append(noiseconcat, newnoise)
        noise = noise[0:len(clean)]

        SNR = random.randint(snr_lower, snr_upper)
        clean_snr, noise_snr, noisy_snr = snr_mixer(clean=clean, noise=noise, snr=SNR)
        return clean_snr, noise_snr, noisy_snr, SNR, noise_name

    def compute_mel(wav, f0_clean, duration=None, basename=None, snr=None):
        if duration is not None:
            # Compute mel-scale spectrogram from preprocessed wav with duration
            mel_spectrogram, energy, clipt = get_mel_and_energy(wav, duration, norm=False)

            # Compute rescaled energy (0 to 1)
            min_, max_ = hp.energy_min, hp.energy_max
            energy_rescaled = (energy-min_)/(max_-min_)
            energy_rescaled = np.clip(energy_rescaled, 0, 1)

            # Compute normalized pitch
            warnings.filterwarnings('error')
            # Get f0 from noisy dataset.
            f0 = get_f0_noisy(wav, duration)
            if (f0==0.).all() or (energy_rescaled==0.).all():
                if (f0==0.).all():
                    print("all zero f0! basename:{} SNR: {}".format(basename, snr))
                else:
                    print("all zero energy! basename:{} SNR: {}".format(basename, snr))
            try:
                f0_norm = utils.speaker_normalization(f0)
            except Warning:
                index_nonzero = (f0 > -1e10)
                mean_f0, std_f0 = np.mean(f0[index_nonzero]), np.std(f0[index_nonzero])
                print('Warning was raised as an exception! basename: {}, mean_f0={} std_f0={}'.format(basename, mean_f0, std_f0)) #, f0, mean_f0, std_f0, f0_file_path)
                f0_norm = utils.speaker_normalization(f0_clean)
            warnings.simplefilter("ignore", DeprecationWarning)

            mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
            f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
            energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)

            # Sanity check
            mel_clean = np.load(os.path.join(out_dir, 'mel_clean', mel_filename))
            assert mel_clean.shape == mel_spectrogram.T.shape, "Computed mel should be the same size with pre-calculated one."
            return (mel_spectrogram.T, f0, f0_norm, energy, energy_rescaled), (mel_filename, f0_filename, energy_filename), clipt
        else:
            # Get mel without any prior info
            print("Oh finally i got here..")
            exit(0)
            mel_spectrogram, _, _ = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav), norm=False)
            return mel_spectrogram.T

    def save_data(data, subdir, filename):
        # Save data to '.npy' file
        np.save(os.path.join(out_dir, subdir, filename), data, allow_pickle=False)

    def save_mel_plot(mel_spectrogram, filename, f0_output, energy_output):
        # Save spectrogram plot to '.png' file
        # _ = utils.plot_spectrogram(mel_spectrogram.T, None, filename)
        utils.plot_data([(mel_spectrogram.T, f0_output, energy_output)], None, filename=filename)

    clipt_audio = set()
    def save_results(wav, f0_clean, duration, basename, basedir, dirtype, noise_name, snr, save_aux):
        # Compute mel from preprocessed wav
        (mel, f0, f0_norm, energy, energy_rescaled), (mel_filename, f0_filename, energy_filename), clipt = compute_mel(wav, f0_clean, duration, basename, snr)
        clipt_audio.add(basename) if clipt else None

        # Save data
        if "a" in dirtype: # ['mel_aug','mel']
            save_data(mel, "mel_aug", mel_filename)
            save_data(f0_norm, "f0_norm_aug", f0_filename)
            save_data(energy_rescaled, "energy_0to1_aug", energy_filename)

        if save_aux:
            # Save mel plot
            save_mel_plot(mel, os.path.join(basedir, "{}_{}_SNRdb_{}_{}.png".format(basename, dirtype, snr, noise_name)), f0, energy)
            audiowrite(wav, hp.sampling_rate, os.path.join(basedir, "{}_{}_SNRdb_{}_{}.wav".format(basename, dirtype, snr, noise_name)), norm=False)

    # Set directory
    train_dir = os.path.join(out_dir, "noise_mixer_results", "train")
    val_dir = os.path.join(out_dir, "noise_mixer_results", "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    # pitch and energy
    os.makedirs(os.path.join(out_dir, 'f0_norm_aug'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'energy_0to1_aug'), exist_ok=True)

    idx_train = 0
    idx_val = 0
    for spker in tqdm(spkers):
        spker_dir = os.path.join(in_dir, spker)
        for dirpath, dirnames, filenames in os.walk(spker_dir):
            for f in filenames:
                if f.endswith(".wav"):
                    basename = f.replace(".wav", "")
                    if basename in unaligned_basenames or (basename not in (cleanbasenames_train+cleanbasenames_val)):
                        continue
                    speaker_id = str(f.split('_')[0])
                    audio_path = os.path.join(dirpath, f)

                    # It's important to reuse pre-calculated data e.g., duration, f0, energy.
                    tg_path = os.path.join(tg_dir, basename.split("_")[0], basename+".TextGrid")
                    clean, _, duration = audioread(audio_path, tg_path=tg_path)
                    f0_clean = get_f0(clean, duration)

                    if basename in cleanbasenames_train:
                        save_aux = idx_train < save_aux_max

                        # Train set (clean speaker)
                        save_results(clean, f0_clean, duration, basename, train_dir, 'tc', "clean", "clean", save_aux)
                        
                        # Augmentation set (clean speaker)
                        _, _, noisy_snr, snr, noise_name = mixer(clean, noisefilenames_train, idx_train)
                        save_results(noisy_snr, f0_clean, duration, basename, train_dir, 'tca', noise_name, snr, save_aux)
                        idx_train += 1

                    elif basename in cleanbasenames_val:
                        save_aux = idx_val < save_aux_max

                        # Validation set (clean speaker)
                        save_results(clean, f0_clean, duration, basename, val_dir, 'vc', "clean", "clean", save_aux)

                        # Augmentation set (clean speaker)
                        _, _, noisy_snr, snr, noise_name = mixer(clean, noisefilenames_val, idx_val)
                        save_results(noisy_snr, f0_clean, duration, basename, val_dir, 'vca', noise_name, snr, save_aux)
                        idx_val += 1

                    else: # Unaligned
                        continue

    ### Sanity check ###
    mel_clean_size = len(os.listdir(os.path.join(out_dir, 'mel_clean')))
    mel_aug_size = len(os.listdir(os.path.join(out_dir, 'mel_aug')))
    print("mel_clean_size:", mel_clean_size)
    print("mel_aug_size:", mel_aug_size)
    assert (len(cleanbasenames_train)+len(cleanbasenames_val)) == mel_aug_size, "Total size should be matched"