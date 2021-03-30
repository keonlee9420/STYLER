import os

# Dataset
dataset = 'VCTK'
data_dir = "/path/to/VCTK-Corpus-92/wav48_silence_trimmed"
noise_dir = "/path/to/wham_noise"

# Speaker Embedding
speaker_embed_dim = 512 # deep-speaker
speaker_embedder_dir = "deepspeaker/pretrained_models/ResCNN_triplet_training_checkpoint_265.h5"

# Version Control
version = "" # set at runtime

# Text
text_cleaners = ['english_cleaners']

# Vocoder
vocoder = 'HiFi-GAN' # ["HiFi-GAN", "MelGAN", "WaveGlow"]
vocoder_speaker = "universal"

# Quantization for F0 and energy
f0_min = 71.0
f0_max = 797.9
energy_min = 0.1
energy_max = 525.43

# Audio and mel
sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024

n_bins = 256

max_wav_value = 32768.0
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0


# STYLER
encoder_layer = 2 # text encoder
encoder_head = 4 # text encoder
encoder_hidden = 256 # text encoder
decoder_layer = 4
decoder_head = 4
decoder_hidden = 256
fft_conv1d_filter_size = 1024
fft_conv1d_kernel_size = (9, 1)
encoder_dropout = 0.2
decoder_dropout = 0.2

style_predictor_filter_size = 256
style_predictor_kernel_size = 3
style_predictor_dropout = 0.5

max_seq_len = 1000

dat_weight = 1
max_mel_len = 1024

va_neck_hidden_t = 4
va_neck_hidden_r = 64
va_neck_hidden_d = 80
va_neck_hidden_p = 64
va_neck_hidden_e = 64

va_enc_dim_r = 256
va_enc_dim_d = 256
va_enc_dim_p = 320
va_enc_dim_e = 320

va_dim_f0 = 257
va_dim_energy = 257
va_chs_grp = 16


# Checkpoints and synthesis path
preprocessed_basedir = "preprocessed"
preprocessed_path = os.path.join(f"./{preprocessed_basedir}/", dataset)
def checkpoint_path(): return os.path.join("./ckpt/", dataset, version)
def synth_path(): return os.path.join("./synth/", dataset, version)
def eval_path(): return os.path.join("./eval/", dataset, version)
def log_path(): return os.path.join("./log/", dataset, version)
def test_path(): return os.path.join("./results/", dataset, version)

# References
ref_audio_dir = "/path/to/ref_audio"
ref_tg_dir = os.path.join(preprocessed_basedir, "ref_audio", "TextGrid")

# Optimizer
batch_size = 16
epochs = 500
n_warm_up_step = 4000
grad_clip_thresh = 1.0
acc_steps = 1

betas = (0.9, 0.98)
eps = 1e-9
weight_decay = 0.


# Log-scaled duration
log_offset = 1.


# Save, log and synthesis
save_step = 10000
synth_step = 1000
eval_step = 1000
eval_size = 2000
log_step = 1000
clear_Time = 20
