import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import argparse
import re
from string import punctuation
from g2p_en import G2p
import time
from styler import STYLER
from dataset import get_processed_data_from_wav
from text import text_to_sequence
import hparams as hp
import utils
import audio as Audio
import shutil
import pyworld as pw
from pysptk import sptk
from scipy.io.wavfile import read
import glob
from deepspeaker import embedding
import pyworld as pw
from data.sentences import sentences

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_audio(mel, energy, f0, f0_norm):
    mel = utils.pad_2D(mel[None])
    f0 = utils.pad_1D(f0[None])
    f0_norm = utils.pad_1D(f0_norm[None])
    energy = utils.pad_1D(energy[None])

    mel_target = torch.from_numpy(mel).float().to(device)
    mel_len = torch.from_numpy(np.array([mel.shape[1]])).long().to(device)
    f0 = torch.from_numpy(f0).float().to(device)
    f0_norm = torch.from_numpy(f0_norm).float().to(device)
    energy = torch.from_numpy(energy).float().to(device)

    return mel_target, mel_len, energy, f0, f0_norm


def preprocess_text(text):
    text = text.rstrip(punctuation)

    g2p = G2p()
    phone = g2p(text)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{' + '}{'.join(phone) + '}'
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    phone = phone.replace('}{', ' ')

    print('|' + phone + '|')
    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    sequence = np.stack([sequence])

    return torch.from_numpy(sequence).long().to(device)


def get_model(checkpoint_path):
    model = nn.DataParallel(STYLER())
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.requires_grad = False
    model.eval()
    return model


def divide_speaker_in_gender(speaker_path):
    speakers = dict()
    with open(speaker_path, encoding='utf-8') as f:
        for line in tqdm(f):
            if "ID" in line: continue
            parts = [p.strip() for p in re.sub(' +', ' ',(line.strip())).split(' ')]
            spk_id, sex = parts[0], parts[2]
            speakers[str(spk_id)] = sex
    return speakers


def model_from_npy(model, basename, tgt_text):
    # Load Data
    speaker_id = basename.split("_")[0]
    src_len = torch.from_numpy(np.array([tgt_text.shape[1]])).to(device)
    mel_path = os.path.join(
            hp.preprocessed_path, "mel_clean", "{}-mel-{}.npy".format(hp.dataset, basename))
    mel = np.load(mel_path)
    f0_norm_path = os.path.join(
            hp.preprocessed_path, "f0_norm", "{}-f0-{}.npy".format(hp.dataset, basename))
    f0_norm = np.load(f0_norm_path)
    energy_path = os.path.join(
            hp.preprocessed_path, "energy_0to1", "{}-energy-{}.npy".format(hp.dataset, basename))
    energy = np.load(energy_path)
    speaker_embed_path = os.path.join(
            hp.preprocessed_path, "spker_embed", "{}-spker_embed-{}.npy".format(hp.dataset, speaker_id))
    speaker_embed = torch.from_numpy(np.load(speaker_embed_path)).to(device)

    mel, mel_len, energy, _, f0_norm = preprocess_audio(mel, energy, f0_norm, f0_norm)

    mel_outputs, mel_postnet_outputs, log_duration_output, f0_output, energy_output, _, _, _, _ = model(
        tgt_text, mel, mel, f0_norm, energy, src_len, mel_len, speaker_embed=speaker_embed)
    return model, src_len, speaker_embed


def prepare_texts(name_list):
    basedir = hp.ref_audio_dir
    text_list = list()
    for name in name_list:
        text = utils.get_transcript(os.path.join(basedir, name+".txt"))
        text = preprocess_text(text)
        text_list.append(text)
    return text_list


def get_encodings(model, current_reference):
    # Encoding
    self_ = model.module.style_modeling
    max_len = self_.max_len
    src_mask = self_.src_mask
    t = self_.text_encoding
    t_neck = self_.text_encoding_neck
    p_down = self_.pitch_encoding # downsampled
    s_down = self_.speaker_encoding_p # downsampled
    p_norm = self_.pitch_linear(p_down)
    p = self_.pitch_linear(p_down + s_down)
    d = self_.duration_encoding
    s = self_.speaker_encoding
    e = self_.energy_encoding
    n = self_.noise_encoding

    return {
        "self_": (self_, current_reference),
        "max_mel_len": (max_len, current_reference),
        "src_mask": (src_mask, current_reference),
        "t": (t, current_reference),
        "t_neck": (t_neck, current_reference),
        "p_down": (p_down, current_reference),
        "s_down": (s_down, current_reference),
        "p_norm": (p_norm, current_reference),
        "p": (p, current_reference),
        "d": (d, current_reference),
        "s": (s, current_reference),
        "e": (e, current_reference),
        "n": (n, current_reference)
    }


def get_encodings_comb(model, step, name, target_name, current_reference):
    encodings_list = list()
    for text in prepare_texts([name, target_name]):
        model, src_len, speaker_embed = model_from_npy(model, name, text)
        encodings = get_encodings(model, current_reference)
        encodings["max_seq_len"] = (src_len, current_reference)
        encodings["speaker_embed"] = (speaker_embed, current_reference)
        encodings_list.append(encodings)
    return encodings_list[0], encodings_list[1]


def get_ref_data(outdir, name):
    audio_path = os.path.join(hp.ref_audio_dir, name+".wav")
    tg_path = os.path.join(hp.ref_tg_dir, name+".TextGrid")
    f0, energy, mel = get_processed_data_from_wav(audio_path, tg_path, noisy_input=False)
    mel, mel_len, energy, f0, _ = preprocess_audio(mel, energy, f0, f0)
    shutil.copy(audio_path, os.path.join(outdir, name + ".wav")) # directly copy reference audio
    mel = mel[0].transpose(0, 1).detach().cpu().numpy()
    f0 = f0[0].detach().cpu().numpy()
    e = energy[0].detach().cpu().numpy()
    return mel, f0, e


def infer(predict, decode, title, t, p, e, d, s, n, src_mask, max_len, speaker_normalized=True, noisy=False):
    t, p, s, e, n, _, f0_output, energy_output, mel_mask = predict(t, p, e, d, s, n, src_mask, max_len, speaker_normalized)
    _, mel_output_postnet = decode((t + p + s + e) if not noisy else (t + p + s + e + n), mel_mask)
    mel_torch = mel_output_postnet.transpose(1, 2).detach()
    mel = mel_output_postnet[0].transpose(0, 1).detach().cpu().numpy()
    f0 = f0_output[0].detach().cpu().numpy()
    e = energy_output[0].detach().cpu().numpy()
    return title, mel_torch, mel, f0, e


def infer_comb(model, enc_comb):
    max_seq_len = enc_comb["max_seq_len"][0]
    max_mel_len = enc_comb["max_mel_len"][0]
    src_mask = enc_comb["src_mask"][0]
    self_ = enc_comb["self_"][0]
    t, ts = enc_comb["t"]
    t_neck, ts = enc_comb["t_neck"]
    d, ds = enc_comb["d"]
    p_down, ps = enc_comb["p_down"]
    e, es = enc_comb["e"]
    s, ss = enc_comb["s"]
    n, ns = enc_comb["n"]
    speaker_embed, ss = enc_comb["speaker_embed"]

    # Target Speaker
    s_down_tgt = self_.style_encoder.speaker_linear_p(speaker_embed).unsqueeze(1).repeat(1, max_seq_len, 1)
    s_tgt = self_.style_encoder.speaker_linear(speaker_embed).unsqueeze(1).repeat(1, max_seq_len, 1)
    p_tgt = self_.pitch_linear(p_down + s_down_tgt)
    self_.speaker_encoding = s_tgt

    # Function (independent from length)
    predict = self_.predict_inference
    decode = model.module.decode

    title, mel_torch, mel, f0, e = infer(predict, decode, f"T{ts}+D{ds}+P{ps}+E{es}+S{ss}", t, t_neck+p_tgt, t_neck+e, t_neck+d, s, n, src_mask, max_mel_len, False)
    return title, f"{ts}{ds}{ps}{es}{ss}", mel_torch, mel, f0, e


def infer_controllability(model, vocoder, step, r1_name, r2_name):
    outdir = os.path.join(hp.test_path(), f"control_r1_{r1_name}_r2_{r2_name}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    titles = []
    comb_arr_titles = []
    mel_torchs = []
    mels = []
    f0s = []
    es = []

    def record(title, arr_str, mel_torch, mel, f0, e):
        comb_arr_titles.append(arr_str)
        mel_torchs.append(mel_torch)
        titles.append(title)
        mels.append(mel)
        f0s.append(f0)
        es.append(e)
    
    # Save references
    for ref_name in [r1_name, r2_name]:
        ref_mel, ref_f0, ref_e = get_ref_data(outdir, ref_name)
        record(ref_name, ref_name, None, ref_mel, ref_f0, ref_e)
    
    # r1
    r1_encodings_r1_text, r1_encodings_r2_text = get_encodings_comb(model, step, r1_name, r2_name, "1")

    # r2
    r2_encodings_r2_text, r2_encodings_r1_text = get_encodings_comb(model, step, r2_name, r1_name,"2")

    def create_enc_comb(comb_arr):
        if comb_arr[0] == 0:
            encodings_list = [r1_encodings_r1_text, r2_encodings_r1_text]
        else:
            encodings_list = [r1_encodings_r2_text, r2_encodings_r2_text]
        enc_comb = {
            "max_seq_len": encodings_list[comb_arr[0]]["max_seq_len"],
            "max_mel_len": encodings_list[comb_arr[0]]["max_mel_len"],
            "src_mask": encodings_list[comb_arr[0]]["src_mask"],
            "self_": encodings_list[comb_arr[0]]["self_"],
            "t": encodings_list[comb_arr[0]]["t"],
            "t_neck": encodings_list[comb_arr[0]]["t_neck"],
            "d": encodings_list[comb_arr[1]]["d"],
            "p_down": encodings_list[comb_arr[2]]["p_down"],
            "e": encodings_list[comb_arr[3]]["e"],
            "s": encodings_list[comb_arr[4]]["s"],
            "speaker_embed": encodings_list[comb_arr[4]]["speaker_embed"],
            "n": encodings_list[comb_arr[0]]["n"],
        }
        return enc_comb

    comb_arrs = set()
    def retrieve_all_comb(n, arr, i): 
        arr_str = "".join([str(n) for n in arr])
        comb_arrs.add(arr_str)
        if i == n:
            return
        arr[i] = 0
        retrieve_all_comb(n, arr, i + 1) 
        arr[i] = 1
        retrieve_all_comb(n, arr, i + 1) 
    retrieve_all_comb(5, [0] * 5 , 0)

    for comb_arr in sorted(list(comb_arrs)):
        # Infer with current combination
        record(*infer_comb(model, create_enc_comb([int(n) for n in comb_arr])))

    # Save data
    for i, (m_torch, m, f0, e, title, comb_arr_title) in enumerate(zip(mel_torchs, mels, f0s, es, titles, comb_arr_titles)):
        utils.plot_data([(m, f0, e)], titles=None, filename=os.path.join(outdir, f'{comb_arr_title}.png'))
        utils.vocoder_infer(m_torch, vocoder, os.path.join(outdir, f'{comb_arr_title}.wav')) if m_torch is not None else None


def infer_inspection(model, vocoder, mel, f0, energy, outdir, sentence):
    titles = ["Reference Spectrogram"]
    mel_torchs = [None]
    mels = [mel[0].transpose(0, 1).detach().cpu().numpy()]
    f0s = [f0[0].detach().cpu().numpy()]
    es = [energy[0].detach().cpu().numpy()]

    def record(title, mel_torch, mel, f0, e):
        mel_torchs.append(mel_torch)
        titles.append(title)
        mels.append(mel)
        f0s.append(f0)
        es.append(e)

    # Encoding
    enc = get_encodings(model, "inspection")
    self_ = enc["self_"][0]
    max_mel_len = enc["max_mel_len"][0]
    src_mask = enc["src_mask"][0]
    t = enc["t"][0]
    t_neck = enc["t_neck"][0]
    p_down = enc["p_down"][0]
    p_norm = enc["p_norm"][0]
    p = enc["p"][0]
    d = enc["d"][0]
    s = enc["s"][0]
    e = enc["e"][0]
    n = enc["n"][0]

    # Function
    predict = self_.predict_inference
    decode = model.module.decode

    print("Inspection...")
    # T+D+P+E+S+N
    record(*infer(predict, decode, "T+D+P+E+S+N", t, t_neck+p, t_neck+e, t_neck+d, s, n, src_mask, max_mel_len, False, True))
    # T+D+P+E+N
    record(*infer(predict, decode, "T+D+P+E+N", t, t_neck+p_norm, t_neck+e, t_neck+d, s, n, src_mask, max_mel_len, True, True))
    # T+D+P+N
    record(*infer(predict, decode, "T+D+P+N", t, t_neck+p_norm, t_neck, t_neck+d, s, n, src_mask, max_mel_len, True, True))
    # T+D+N
    record(*infer(predict, decode, "T+D+N", t, t_neck, t_neck, t_neck+d, s, n, src_mask, max_mel_len, True, True))
    # T+N
    record(*infer(predict, decode, "T+N", t, t_neck, t_neck, t_neck, s, n, src_mask, max_mel_len, True, True))
    # T
    record(*infer(predict, decode, "T", t, t_neck, t_neck, t_neck, s, n, src_mask, max_mel_len, True))
    # T+D
    record(*infer(predict, decode, "T+D", t, t_neck, t_neck, t_neck+d, s, n, src_mask, max_mel_len, True))
    # T+D+P
    record(*infer(predict, decode, "T+D+P", t, t_neck+p_norm, t_neck, t_neck+d, s, n, src_mask, max_mel_len, True))
    # T+D+P+E
    record(*infer(predict, decode, "T+D+P+E", t, t_neck+p_norm, t_neck+e, t_neck+d, s, n, src_mask, max_mel_len, True))
    # T+D+P+E+S
    record(*infer(predict, decode, "T+D+P+E+S", t, t_neck+p, t_neck+e, t_neck+d, s, n, src_mask, max_mel_len,  False))
    print("Done!")

    # Save data
    for i, (m_torch, m, f0, e, title) in enumerate(zip(mel_torchs, mels, f0s, es, titles)):
        utils.plot_data([(m, f0, e)], [title], filename=os.path.join(outdir, '{}_{}_{}_{}.png'.format('i', hp.vocoder, sentence[:10], "inspect{}".format(i))))
        utils.vocoder_infer(m_torch, vocoder, os.path.join(outdir, '{}_{}_{}_{}.wav'.format('i', hp.vocoder, sentence[:10], "inspect{}".format(i)))) if m_torch is not None else None


def synthesize(outdir, model, vocoder, text, sentence, speaker_embed, speaker_id, inspection, mel_raw, mel_len, f0, f0_norm, energy, duration_control=1.0, pitch_control=1.0, energy_control=1.0):
    sentence = sentence[:100]  # long filename will result in OS Error

    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)
    mel_outputs, mel_postnet_outputs, log_duration_output, f0_output, energy_output, _, _, _, _ = model(
        text, mel_raw, mel_raw, f0_norm, energy, src_len, mel_len, speaker_embed=speaker_embed, d_control=duration_control, p_control=pitch_control, e_control=energy_control)
    mel, mel_postnet = mel_outputs[0], mel_postnet_outputs[0]
    mel_noisy, mel_postnet_noisy = mel_outputs[1], mel_postnet_outputs[1]

    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    mel_postnet_noisy_torch = mel_postnet_noisy.transpose(1, 2).detach()
    mel = mel[0].cpu().transpose(0, 1).detach()
    mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()
    mel_noisy = mel_noisy[0].cpu().transpose(0, 1).detach()
    mel_postnet_noisy = mel_postnet_noisy[0].cpu().transpose(0, 1).detach()
    f0_output = f0_output[0].detach().cpu().numpy()
    energy_output = energy_output[0].detach().cpu().numpy()

    # Save clean
    # Audio.tools.inv_mel_spec(mel_postnet, os.path.join(
    #     outdir, '{}_{}_{}.wav'.format("c", hp.vocoder, sentence)))
    utils.vocoder_infer(mel_postnet_torch, vocoder, os.path.join(
        outdir, '{}_{}_{}.wav'.format("c", hp.vocoder, sentence)))
    # Model mel prediction
    utils.plot_data([(mel_postnet.numpy(), f0_output, energy_output)], [
                    'Synthesized Spectrogram Clean'], filename=os.path.join(outdir, '{}_{}_{}.png'.format("c", hp.vocoder, sentence)))

    # Save noisy
    # Audio.tools.inv_mel_spec(mel_postnet_noisy, os.path.join(
    #     outdir, '{}_{}_{}.wav'.format("n", hp.vocoder, sentence)))
    utils.vocoder_infer(mel_postnet_noisy_torch, vocoder, os.path.join(
        outdir, '{}_{}_{}.wav'.format("n", hp.vocoder, sentence)))
    # Model mel prediction
    utils.plot_data([(mel_postnet_noisy.numpy(), f0_output, energy_output)], [
                    'Synthesized Spectrogram Noisy'], filename=os.path.join(outdir, '{}_{}_{}.png'.format("n", hp.vocoder, sentence)))
    # utils.plot_spectrogram(mel_postnet_noisy.numpy(), "Synthesized Spectrogram Noisy", filename=os.path.join(outdir, '{}_{}_{}.png'.format("n", hp.vocoder, sentence)))

    # Inspection
    if inspection:
        energy = energy*(hp.energy_max-hp.energy_min) + hp.energy_min
        infer_inspection(model, vocoder, mel_raw, f0, energy, outdir, sentence)

    # # Model duration prediction
    # log_duration_output = log_duration_output[0].detach().cpu() # [seg_len]
    # log_duration_output = torch.clamp(torch.round(torch.exp(log_duration_output)-hp.log_offset), min=0).int()
    # model_duration = utils.get_alignment_2D(log_duration_output).T # [seg_len, mel_len]
    # model_duration = utils.plot_alignment([model_duration], filename=os.path.join(outdir, '{}_{}_{}.png'.format("d", hp.vocoder, sentence)))


def synthesize_with_reference(idx_info, name, noisy_input, audio_path, tg_path, speaker_id, inspection):
    global model, vocoder, step
    start_time = time.perf_counter()

    # Prepare Reference Data
    if speaker_id is not None:
        spker_embed_path = os.path.join(
                hp.preprocessed_path, "spker_embed", "{}-spker_embed-{}.npy".format(hp.dataset, speaker_id))
        speaker_embed = torch.from_numpy(np.load(spker_embed_path)).to(device)
    else:
        try:
            # VCTK fileformat
            speaker_id = name.split("_")[0]
            spker_embed_path = os.path.join(
                hp.preprocessed_path, "spker_embed", "{}-spker_embed-{}.npy".format(hp.dataset, speaker_id))
            speaker_embed = torch.from_numpy(np.load(spker_embed_path)).to(device)
        except:
            # General cases
            speaker_id = None
            speaker_embed = torch.from_numpy(embedding.predict_embedding(speaker_embedder, audio_path))

    # Outdir
    outdir = os.path.join(hp.test_path(), "{}_by_{}_{}".format(name, speaker_id, step))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    text = utils.get_transcript(os.path.join(audio_path.replace(".wav", ".txt")))
    if not os.path.isfile(tg_path):
        tg_path = "NO TextGrid"
        _, wav = read(audio_path)
        if noisy_input:
            f0 = sptk.rapt(wav.astype(np.float32)*hp.max_wav_value, hp.sampling_rate, hp.encoder_hidden, min=hp.f0_min, max=hp.f0_max, otype=2) # log f0
            f0 = np.exp(f0)
        else:
            f0, _ = pw.dio(wav.astype(np.float64), hp.sampling_rate, frame_period=hp.hop_length/hp.sampling_rate*1000)
        mel, energy, _ = Audio.tools.get_mel_from_wav(torch.FloatTensor(np.array(wav)))
        mel = mel.T.numpy().astype(np.float32)
        energy = energy.numpy().astype(np.float32)
        utils.plot_data([(mel.T, f0, energy)], [
                    'Reference Spectrogram'], filename=os.path.join(outdir, '{}_{}_{}.png'.format("Reference", name, text[:100])))
    else:
        f0, energy, mel = get_processed_data_from_wav(audio_path, tg_path, noisy_input)
        utils.plot_data([(mel.T, f0, energy)], [
                    'Reference Spectrogram'], filename=os.path.join(outdir, '{}_{}_{}.png'.format("Reference", name, text[:100])))

    # Prepare Audio Inputs
    energy = (energy-hp.energy_min)/(hp.energy_max-hp.energy_min)
    f0_norm = utils.speaker_normalization(f0)
    mel, mel_len, energy, f0, f0_norm = preprocess_audio(mel, energy, f0, f0_norm)

    print("\n\n---------------- [{}/{}]: {} ----------------".format(idx_info[0]+1, idx_info[1],audio_path.split('/')[-1]))
    print('Audio Path:', audio_path)
    print('TextGrid Path:', tg_path)
    print('Speaker ID:', speaker_id)

    # Synthesize
    success = 0
    for sentence in sentences:
        text = preprocess_text(sentence)
        synthesize(outdir, model, vocoder, text, sentence, speaker_embed, speaker_id, inspection, mel, mel_len, f0, f0_norm, energy, args.duration_control, args.pitch_control, args.energy_control)
        success += 1
    print("Synthesized {} out of {} in {:.3f}s".format(success, len(sentences), time.perf_counter()-start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default="ckpt/default/checkpoint_300000.pth.tar")
    parser.add_argument('--cont', action='store_true', default=False)
    parser.add_argument('--r1', type=str, default="p323_229")
    parser.add_argument('--r2', type=str, default="p259_284")
    parser.add_argument('--ref_dir', type=str, default=hp.ref_audio_dir)
    parser.add_argument('--ref_name', type=str, default="")
    parser.add_argument('--speaker_id', type=str, default=None)
    parser.add_argument('--noisy_input', action='store_true', default=False)
    parser.add_argument('--inspection', action='store_true', default=False)
    parser.add_argument('--duration_control', type=float, default=1.0)
    parser.add_argument('--pitch_control', type=float, default=1.0)
    parser.add_argument('--energy_control', type=float, default=1.0)
    args = parser.parse_args()
    step = args.ckpt_path.split("/")[-1].split(".")[0].split("_")[1]

    # Version Control
    hp.version = args.ckpt_path.split("/")[-2]

    # Set Reference Directory
    hp.ref_audio_dir = args.ref_dir
    hp.ref_tg_dir = os.path.join(hp.preprocessed_basedir, args.ref_dir.split("/")[-1], "TextGrid")

    model = get_model(args.ckpt_path).to(device)
    vocoder = utils.get_vocoder()
    speaker_embedder = embedding.build_model(hp.speaker_embedder_dir)

    with torch.no_grad():
        start_time_total = time.perf_counter()
        idx = 0
        if args.cont:
            infer_controllability(model, vocoder, step, args.r1, args.r2)
        elif args.ref_name != "":
            print("\nSingle-inference")
            synthesize_with_reference((idx, 1), args.ref_name, args.noisy_input,
                                      os.path.join(hp.ref_audio_dir, args.ref_name+".wav"),
                                      os.path.join(hp.ref_tg_dir, reference+".TextGrid"),
                                      args.speaker_id, args.inspection)
        else:
            print("\nMulti-inference")
            references = [str(ref_dir).split('/')[-1].replace('.wav','') for ref_dir in glob.glob(os.path.join(hp.ref_audio_dir, '*.wav'))]
            print(references)
            for reference in references:
                synthesize_with_reference((idx, len(references)), reference, args.noisy_input,
                                      os.path.join(hp.ref_audio_dir, reference+".wav"),
                                      os.path.join(hp.ref_tg_dir, reference+".TextGrid"),
                                      args.speaker_id, args.inspection)
                idx+=1
    print("All done in {:.3f}s".format(time.perf_counter()-start_time_total))