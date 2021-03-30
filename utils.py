import hparams as hp
import text
from scipy.io import wavfile
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import hifigan
import json
import warnings
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_transcript(txt_path):
    with open(os.path.join(txt_path)) as f:
        return f.readline().strip()


def mfa(in_dir, out_dir, mfa_path="./montreal-forced-aligner"): 
    """
    See https://github.com/ga642381/STYLER/blob/5404756a97d7ce69e7c7327dd4c38dec5dfbac8c/preprocess.py#L102
    """

    mfa_out_dir = os.path.join(out_dir, "TextGrid")
    mfa_bin_path = os.path.join(mfa_path, "bin", "mfa_align")
    mfa_pretrain_path = os.path.join(mfa_path, "pretrained_models", "librispeech-lexicon.txt")
    cmd = f"{mfa_bin_path} {in_dir} {mfa_pretrain_path} english {mfa_out_dir} -j 8"
    print(cmd)
    os.system(cmd)

    return mfa_out_dir


def get_alignment(tier):
    sil_phones = ['sil', 'sp', 'spn']

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trimming leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s
        if p not in sil_phones:
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            phones.append(p)
        durations.append(int(np.round(
            e*hp.sampling_rate/hp.hop_length)-np.round(s*hp.sampling_rate/hp.hop_length)))

    # Trimming tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]

    return phones, durations, start_time, end_time


def get_alignment_2D(duration_predictor_output):
    L = duration_predictor_output.size(0)
    expand_max_len = torch.max(duration_predictor_output).item()
    alignment = torch.zeros(L*expand_max_len, L)

    count = 0
    for i in range(L):
        for j in range(int(duration_predictor_output[i])):
            alignment[count+j][i] = 1.
        count = count + int(duration_predictor_output[i])

    return alignment # [mel_len, seg_len]


def process_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        text = []
        name = []
        for line in f.readlines():
            n, t = line.strip('\n').split('|')
            name.append(n)
            text.append(t)
        return name, text


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def plot_data(data, titles, filename):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax, offset=0):
        ax = fig.add_axes(old_ax.get_position(), anchor='W')
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        spectrogram, pitch, energy = data[i]
        axes[i][0].imshow(spectrogram, origin='lower')
        axes[i][0].set_aspect(2.5, adjustable='box')
        axes[i][0].set_ylim(0, hp.n_mel_channels)
        axes[i][0].set_title(titles[i], fontsize='medium')
        axes[i][0].tick_params(labelsize='x-small',
                               left=False, labelleft=False)
        axes[i][0].set_anchor('W')

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color='tomato')
        ax1.set_xlim(0, spectrogram.shape[1])
        ax1.set_ylim(0, hp.f0_max)
        ax1.set_ylabel('F0', color='tomato')
        ax1.tick_params(labelsize='x-small', colors='tomato',
                        bottom=False, labelbottom=False)

        ax2 = add_axis(fig, axes[i][0], 1.2)
        ax2.plot(energy, color='darkviolet')
        ax2.set_xlim(0, spectrogram.shape[1])
        ax2.set_ylim(hp.energy_min, hp.energy_max)
        ax2.set_ylabel('Energy', color='darkviolet')
        ax2.yaxis.set_label_position('right')
        ax2.tick_params(labelsize='x-small', colors='darkviolet', bottom=False,
                        labelbottom=False, left=False, labelleft=False, right=True, labelright=True)
    
    # Save to filename
    plt.savefig(filename, dpi=200)

    # Save to numpy
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)

    plt.close()

    return data


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_spectrogram(spectrogram, title, filename):
    fig, ax = plt.subplots()

    ax.imshow(spectrogram, origin='lower')
    ax.set_aspect(2.5, adjustable='box')
    ax.set_ylim(0, hp.n_mel_channels)
    ax.set_title(title, fontsize='medium') if title is not None else None
    ax.tick_params(labelsize='x-small',
                            left=False, labelleft=False)
    ax.set_anchor('W')

    # Save to filename
    plt.savefig(filename, bbox_inches='tight', dpi=200)

    # Save to numpy
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)

    plt.close()
    return data


def plot_alignment(alignments, infos=None, filename=None, titles=None):
    if len(alignments) <= 2:
        ph, pw = 1, len(alignments)
    else:
        ph, pw = 2, (len(alignments)+1)//2

    fig, axes = plt.subplots(ph, pw, squeeze=False)

    if titles is None:
        titles = [None for i in range(len(alignments))]

    for h in range(ph):
        for w in range(pw):
            alignment = alignments[h*pw+w]
            im = axes[h][w].imshow(alignment, aspect='auto', origin='lower',
                    interpolation='none')
            fig.colorbar(im, ax=axes[h][w])
            axes[h][w].set_title(titles[h*pw+w], fontsize='medium')
            axes[h][w].tick_params(labelsize='x-small')
            if w == 0:
                axes[h][w].set_ylabel('Encoder timestep')
            if h == ph-1:
                xlabel = 'Decoder timestep'
                if infos is not None:
                    xlabel += '\n\n' + infos[h*pw+w]
                axes[h][w].set_xlabel(xlabel)
            axes[h][w].set_anchor('W')
    plt.tight_layout()

    # Save to filename
    if filename is not None:
        plt.savefig(filename, dpi=200)

    # Save to numpy
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)

    plt.close()

    return data


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(
        0).expand(batch_size, -1).to(device)
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))

    return mask


def get_vocoder():
    name = hp.vocoder
    speaker = hp.vocoder_speaker

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
    elif name == "WaveGlow":
        vocoder = torch.hub.load(
            'nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
        vocoder = vocoder.remove_weightnorm(vocoder)
        vocoder.eval()
        for m in vocoder.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')
        vocoder.to(device)

    return vocoder


def vocoder_infer(mel, vocoder, path):
    name = hp.vocoder
    with torch.no_grad():
        if name == "MelGAN":
            wav = vocoder.inverse(mel / np.log(10))
        elif name == "HiFi-GAN":
            wav = vocoder(mel).squeeze(1)
        elif name == "WaveGlow":
            wav = vocoder.infer(mel, sigma=1.0)

    wav = (
        wav.squeeze().cpu().numpy()
        * hp.max_wav_value
    ).astype("int16")

    wavfile.write(path, hp.sampling_rate, wav)

    return wav


def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def get_scale(src, tgt):
    return [src // tgt + (1 if x < src % tgt else 0) for x in range (tgt)]


def mel_calibrator(mel, mel_len, seq_len):
    """
    mel --- [batch, mel_len, mel_hidden]
    mel_len --- [batch,]
    seq_len --- [batch,]
    scaled_mel --- [batch, src_len, mel_hidden]
    """
    batch = []
    for b in range(mel_len.shape[0]):
        ml, sl = int(mel_len[b].item()), int(seq_len[b].item())
        m = mel[b, :ml]
        if sl == ml:
            batch.append(m)
            continue
        elif ml > sl: 
            # Compression
            split_size = get_scale(ml, sl) # len == sl
            m = nn.utils.rnn.pad_sequence(torch.split(m, split_size, dim=0)) # [unit_len, seq_len, mel_hidden]
            m = torch.div(torch.sum(m, dim=0), torch.tensor(split_size, device=m.device).unsqueeze(-1)) # [seq_len, mel_hidden]
            batch.append(m)
        else: 
            # Expansions
            repeat_size = get_scale(sl, ml) # len == ml
            m = torch.repeat_interleave(m, torch.tensor(repeat_size, device=m.device), dim=0) # [seq_len, mel_hidden]
            batch.append(m)

    # Re-padding
    scaled_mel = pad(batch)

    return scaled_mel


def speaker_normalization(f0):
    f0 = f0.astype(float).copy()
    index_nonzero = (f0 > -1e10)
    mean_f0, std_f0 = np.mean(f0[index_nonzero]), np.std(f0[index_nonzero])
    # f0 is logf0
    # f0 = np.log(f0)
    #index_nonzero = f0 != 0
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    return f0 # np.exp(f0)


def f0_normalization(f0):
    warnings.filterwarnings('error')
    try:
        f0_norm = speaker_normalization(f0)
    except Warning:
        f0_norm = np.zeros_like(f0)
    warnings.resetwarnings()
    return f0_norm


def energy_rescaling(energy):
    min_, max_ = hp.energy_min, hp.energy_max
    energy_rescaled = (energy-min_)/(max_-min_)
    energy_rescaled = np.clip(energy_rescaled, 0, 1)
    return energy_rescaled


def quantize_1D_torch(x, num_bins=256):
    # x is logf0
    B = x.size(0)
    x = x.view(-1).clone()
    uv = (x<=0)
    x[uv] = 0
    assert (x >= 0).all() and (x <= 1).all()
    x = torch.round(x * (num_bins-1))
    x = x + 1
    x[uv] = 0
    enc = torch.zeros((x.size(0), num_bins+1), device=x.device)
    enc[torch.arange(x.size(0)), x.long()] = 1
    return enc.view(B, -1, num_bins+1), x.view(B, -1).long()