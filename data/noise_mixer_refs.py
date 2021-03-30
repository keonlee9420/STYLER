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


def basenames_and_transcripts(in_dir):
    basenames = list()
    transcripts = list()
    for ref_path in glob.glob(os.path.join(in_dir, '*.wav')):
        basename = ref_path.split("/")[-1].replace(".wav","")
        text = utils.get_transcript(ref_path.replace(".wav", ".txt"))
        basenames.append(basename)
        transcripts.append(text)
        print(basename, text)
    return basenames, transcripts


def build_from_path(in_dir,
                noise_dir=hp.noise_dir, 
                snr_lower=5, 
                snr_upper=25,
                silence_length=0.2,
                save_aux_max=40,
                noise_speaker_rate=0.5):
    out_dir = in_dir+"_noisy"

    ref_dir_name = in_dir.split("/")[-1]
    target_refs, target_refs_transcript = basenames_and_transcripts(in_dir)
    print("Total target size : {}".format(len(target_refs)))

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

    def compute_mel(wav, f0_clean):
        # Get mel without any prior info
        mel_spectrogram, energy, _ = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav), norm=False)
        return mel_spectrogram.T, energy

    def save_mel_plot(mel_spectrogram, filename, f0_output, energy_output):
        # Save spectrogram plot to '.png' file
        # _ = utils.plot_spectrogram(mel_spectrogram.T, None, filename)
        utils.plot_data([(mel_spectrogram.T, f0_output, energy_output)], None, filename=filename)

    # Set directory
    os.makedirs(out_dir, exist_ok=True)

    idx_target = 0
    filelist_list = list()
    for basename, text in zip(target_refs, target_refs_transcript):
        audio_path = os.path.join(in_dir, basename+".wav")

        clean, _, _ = audioread(audio_path)

        # Save clean mel
        f0_clean = get_f0(clean)
        mel_clean, energy_clean = compute_mel(clean, f0_clean)
        save_mel_plot(mel_clean, os.path.join(out_dir, basename+"_org.png"), f0_clean, energy_clean)

        # Mix and save
        _, _, noisy_snr, snr, noise_name = mixer(clean, noisefilenames_train, idx_target)
        audiowrite(noisy_snr, hp.sampling_rate, os.path.join(out_dir, basename+".wav"), norm=False)
        f0_noisy = get_f0_noisy(noisy_snr)
        mel_noisy, energy_noisy = compute_mel(noisy_snr, f0_noisy)
        save_mel_plot(mel_noisy, os.path.join(out_dir, basename+"_noisy.png"), f0_noisy, energy_noisy)

        filelist_list.append("|".join([basename, text, str(snr), noise_name]))
        idx_target += 1

    ### Write Filelist ###
    with open(os.path.join(hp.preprocessed_basedir, ref_dir_name, '{}_noisy.txt'.format(ref_dir_name)), 'w', encoding='utf-8') as f:
        print("Total saved filelist elements:", len(filelist_list))
        for row in filelist_list:
            f.write(str(row)+'\n')

    ### Sanity check ###
    assert len(target_refs) == len(filelist_list), "Total size should be matched"