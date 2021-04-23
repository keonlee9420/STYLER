import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd  import Function
from collections import OrderedDict
import numpy as np
import copy

import hparams as hp
import utils

import transformer.Constants as Constants
from transformer.Models import Encoder
from transformer.Layers import ConvNorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class AugmentationClassifier(nn.Module):
    """ Simple augmentation classifier """

    def __init__(self, input_dim=hp.encoder_hidden):
        super(AugmentationClassifier, self).__init__()
        self.grl = GradientReversalLayer()
        self.hidden = hp.encoder_hidden
        self.classifier = nn.Sequential(OrderedDict([
            ('d_fc1', nn.Linear(input_dim, self.hidden)),
            ('d_bn1', nn.LayerNorm(self.hidden)),
            ('d_relu1', nn.ReLU()),
            ('d_fc2', nn.Linear(self.hidden, 2)),
            ('d_softmax', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, x):
        # GRL
        rev_x = self.grl(x)
        # Calculate augmentation posterior
        score = self.classifier(rev_x)
        if len(score.size()) > 2:
            score = score.mean(dim=1)
        return score # [batch, 2]


class RevGrad(Function):
    """
    A gradient reversal layer.
    This layer has no parameters, and simply reverses the gradient in the backward pass.
    See https://www.codetd.com/en/article/11984164, https://github.com/janfreyberg/pytorch-revgrad
    """
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__()

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return RevGrad.apply(input_, self._alpha)


class AudioEncoder(nn.Module):
    """Encoder for audio-related style factors.
    """
    def __init__(self):
        super().__init__()

        self.va_chs_grp = hp.va_chs_grp
        self.n_mel_channels = hp.n_mel_channels
        self.va_dim_energy = hp.va_dim_energy
        self.va_dim_f0 = hp.va_dim_f0
        self.va_enc_dim_d = hp.va_enc_dim_d
        self.va_enc_dim_e = hp.va_enc_dim_e
        self.va_enc_dim_p = hp.va_enc_dim_p
        self.va_enc_dim_r = hp.va_enc_dim_r
        self.va_neck_hidden_d = hp.va_neck_hidden_d
        self.va_neck_hidden_e = hp.va_neck_hidden_e
        self.va_neck_hidden_p = hp.va_neck_hidden_p
        self.va_neck_hidden_r = hp.va_neck_hidden_r
        
        # convolutions for duration
        n_layers = 3
        convolutions = []
        for i in range(n_layers):
            conv_layer = nn.Sequential(
                ConvNorm(self.n_mel_channels if i==0 else self.va_enc_dim_d,
                         self.va_enc_dim_d,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.va_enc_dim_d//self.va_chs_grp, self.va_enc_dim_d))
            convolutions.append(conv_layer)
        self.convolutions_1 = nn.ModuleList(convolutions)
        
        self.lstm_1 = nn.LSTM(self.va_enc_dim_d, self.va_neck_hidden_d, 2, batch_first=True, bidirectional=True)

        # convolutions for f0
        convolutions = []
        for i in range(n_layers):
            conv_layer = nn.Sequential(
                ConvNorm(self.va_dim_f0 if i==0 else self.va_enc_dim_p,
                         self.va_enc_dim_p,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.va_enc_dim_p//self.va_chs_grp, self.va_enc_dim_p))
            convolutions.append(conv_layer)
        self.convolutions_2 = nn.ModuleList(convolutions)
        
        self.lstm_2 = nn.LSTM(self.va_enc_dim_p, self.va_neck_hidden_p, 2, batch_first=True, bidirectional=True)
        
        # convolutions for energy
        convolutions = []
        for i in range(n_layers):
            conv_layer = nn.Sequential(
                ConvNorm(self.va_dim_energy if i==0 else self.va_enc_dim_e,
                         self.va_enc_dim_e,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.va_enc_dim_e//self.va_chs_grp, self.va_enc_dim_e))
            convolutions.append(conv_layer)
        self.convolutions_3 = nn.ModuleList(convolutions)
        
        self.lstm_3 = nn.LSTM(self.va_enc_dim_e, self.va_neck_hidden_e, 2, batch_first=True, bidirectional=True)

        # convolutions for residual
        convolutions = []
        for i in range(n_layers):
            conv_layer = nn.Sequential(
                ConvNorm(self.n_mel_channels if i==0 else self.va_enc_dim_r,
                         self.va_enc_dim_r,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.va_enc_dim_r//self.va_chs_grp, self.va_enc_dim_r))
            convolutions.append(conv_layer)
        self.convolutions_4 = nn.ModuleList(convolutions)
        
        self.lstm_4 = nn.LSTM(self.va_enc_dim_r, self.va_neck_hidden_r, 2, batch_first=True, bidirectional=True)

    def forward(self, cat, len_org, seq_len, mask):

        d, f0, e, r = torch.split(cat, [self.n_mel_channels, self.va_dim_f0, self.va_dim_energy, self.n_mel_channels], dim=1)

        for i, (conv_1, conv_2, conv_3, conv_4) in enumerate(zip(self.convolutions_1, self.convolutions_2, self.convolutions_3, self.convolutions_4)):
            d = F.relu(conv_1(d))
            f0 = F.relu(conv_2(f0))
            e = F.relu(conv_3(e))
            r = F.relu(conv_4(r))

        cat = torch.cat((d, f0, e, r), dim=1).transpose(1, 2)
        cat = utils.mel_calibrator(cat, len_org, seq_len)

        d, f0, e, r = torch.split(cat, [self.va_enc_dim_d, self.va_enc_dim_p, self.va_enc_dim_e, self.va_enc_dim_r], dim=-1)

        d = self.lstm_1(d)[0]
        f0 = self.lstm_2(f0)[0]
        e = self.lstm_3(e)[0]
        r = self.lstm_4(r)[0]
        
        d_forward = d[:, :, :self.va_neck_hidden_d]
        d_backward = d[:, :, self.va_neck_hidden_d:]
        
        f0_forward = f0[:, :, :self.va_neck_hidden_p]
        f0_backward = f0[:, :, self.va_neck_hidden_p:]
        
        e_forward = e[:, :, :self.va_neck_hidden_e]
        e_backward = e[:, :, self.va_neck_hidden_e:]
        
        r_forward = r[:, :, :self.va_neck_hidden_r]
        r_backward = r[:, :, self.va_neck_hidden_r:]

        duration_encoding = torch.cat((d_forward, d_backward), dim=-1)
        f0_encoding = torch.cat((f0_forward, f0_backward), dim=-1)
        energy_encoding = torch.cat((e_forward, e_backward), dim=-1)
        noise_encoding = torch.cat((r_forward, r_backward), dim=-1)

        return duration_encoding, f0_encoding, energy_encoding, noise_encoding


class StyleEncoder(nn.Module):
    """ Style Encoder """

    def __init__(self):
        super(StyleEncoder, self).__init__()
        self.text_encoder = Encoder()
        self.audio_encoder = AudioEncoder()
        self.text_linear_down = nn.Sequential(nn.Linear(hp.encoder_hidden, hp.va_neck_hidden_t),
                                            nn.ReLU())
        self.speaker_linear_p = nn.Sequential(nn.Linear(hp.speaker_embed_dim, hp.va_neck_hidden_p*2),
                                            nn.ReLU())
        self.speaker_linear = nn.Sequential(nn.Linear(hp.speaker_embed_dim, hp.encoder_hidden),
                                            nn.ReLU())

    def encoder_input_cat(self, mel_target, p_norm, e_input, mel_aug):
        p_norm_quantized = utils.quantize_1D_torch(p_norm.unsqueeze(-1))[0]
        e_input_quantized = utils.quantize_1D_torch(e_input.unsqueeze(-1))[0]
        enc_cat = torch.cat((mel_target, p_norm_quantized, e_input_quantized, mel_aug), dim=-1)
        enc_cat = enc_cat.transpose(2,1)
        return enc_cat

    def forward(self, text, speaker_embed, mel_target, p_norm, e_input, mel_aug, mel_len, src_len, src_mask):

        # Encoding
        text_encoding = self.text_encoder(text, src_mask)
        text_encoding_neck = self.text_linear_down(text_encoding)
        speaker_encoding_p = self.speaker_linear_p(speaker_embed)
        speaker_encoding = self.speaker_linear(speaker_embed)
        enc_cat = self.encoder_input_cat(mel_target, p_norm, e_input, mel_aug)
        duration_encoding, pitch_encoding, energy_encoding, noise_encoding = self.audio_encoder(enc_cat, mel_len, src_len, mask=None)

        return text_encoding, text_encoding_neck, speaker_encoding_p, speaker_encoding, duration_encoding, pitch_encoding, energy_encoding, noise_encoding


class StyleModeling(nn.Module):
    """ Style Modeling """

    def __init__(self):
        super(StyleModeling, self).__init__()

        self.style_encoder = StyleEncoder()

        self.augmentation_classifier_d = AugmentationClassifier(input_dim=hp.va_neck_hidden_d*2)
        self.augmentation_classifier_p = AugmentationClassifier(input_dim=hp.va_neck_hidden_p*2)
        self.augmentation_classifier_e = AugmentationClassifier(input_dim=hp.va_neck_hidden_e*2)

        self.duration_linear = nn.Sequential(nn.Linear(hp.va_neck_hidden_d*2, hp.encoder_hidden),
                                            nn.ReLU(),
                                            nn.Linear(hp.encoder_hidden, hp.encoder_hidden),
                                            nn.ReLU())
        self.pitch_norm_linear = nn.Sequential(nn.Linear(hp.va_neck_hidden_p*2, hp.encoder_hidden),
                                            nn.ReLU(),
                                            nn.Linear(hp.encoder_hidden, hp.encoder_hidden),
                                            nn.ReLU())
        self.pitch_linear = nn.Sequential(nn.Linear(hp.va_neck_hidden_p*2, hp.encoder_hidden),
                                            nn.ReLU(),
                                            nn.Linear(hp.encoder_hidden, hp.encoder_hidden),
                                            nn.ReLU())
        self.energy_linear = nn.Sequential(nn.Linear(hp.va_neck_hidden_e*2, hp.encoder_hidden),
                                            nn.ReLU(),
                                            nn.Linear(hp.encoder_hidden, hp.encoder_hidden),
                                            nn.ReLU())
        self.residual_linear = nn.Sequential(nn.Linear(hp.va_neck_hidden_r*2, hp.encoder_hidden),
                                            nn.ReLU(),
                                            nn.Linear(hp.encoder_hidden, hp.encoder_hidden),
                                            nn.ReLU())
        self.text_linear_up = nn.Sequential(nn.Linear(hp.va_neck_hidden_t, hp.encoder_hidden),
                                            nn.ReLU())

        self.duration_predictor = StylePredictor()
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = StylePredictor()
        self.energy_predictor = StylePredictor()

        self.pitch_bins = nn.Parameter(torch.exp(torch.linspace(
            np.log(hp.f0_min), np.log(hp.f0_max), hp.n_bins-1)), requires_grad=False)
        self.energy_bins = nn.Parameter(torch.linspace(
            hp.energy_min, hp.energy_max, hp.n_bins-1), requires_grad=False)
        self.pitch_embedding = nn.Embedding(hp.n_bins, hp.encoder_hidden)
        self.energy_embedding = nn.Embedding(hp.n_bins, hp.encoder_hidden)

    def predict_inference(self, text_encoding, pitch_encoding, energy_encoding, duration_encoding, speaker_encoding, noise_encoding, src_mask, max_len, speaker_normalized=True, d_control=1.0, p_control=1.0, e_control=1.0):
        encodings_cat = torch.cat((text_encoding, pitch_encoding, speaker_encoding, energy_encoding, noise_encoding), dim=-1)

        # Duration
        log_duration_prediction = self.duration_predictor(duration_encoding, src_mask) # [batch_size, src_len]
        duration_rounded = torch.clamp(
            (torch.round(torch.exp(log_duration_prediction)-hp.log_offset)*d_control), min=0)
        encodings_cat, mel_len = self.length_regulator(encodings_cat, duration_rounded, max_len)
        mel_mask = utils.get_mask_from_lengths(mel_len)

        text_encoding, pitch_encoding, speaker_encoding, energy_encoding, noise_encoding = torch.split(encodings_cat, hp.encoder_hidden, dim=-1)

        # Energy
        energy_prediction = self.energy_predictor(energy_encoding, mel_mask)
        energy_prediction = energy_prediction*e_control
        energy_embedding = self.energy_embedding(
            torch.bucketize(energy_prediction, self.energy_bins))

        # Pitch
        pitch_prediction = self.pitch_predictor(pitch_encoding if speaker_normalized else (pitch_encoding + speaker_encoding), mel_mask)
        pitch_prediction = pitch_prediction*p_control
        pitch_embedding = self.pitch_embedding(
            torch.bucketize(pitch_prediction, self.pitch_bins))
        
        return text_encoding, pitch_embedding, speaker_encoding, energy_embedding, noise_encoding, log_duration_prediction, pitch_prediction, energy_prediction, mel_mask

    def forward(self, text, speaker_embed, mel_target, mel_aug, p_norm, e_input, src_len, mel_len, src_mask, mel_mask=None, duration_target=None, pitch_target=None, energy_target=None, max_len=None, d_control=1.0, p_control=1.0, e_control=1.0):

        # Encoding
        text_encoding, text_encoding_neck, speaker_encoding_p, speaker_encoding, duration_encoding, pitch_encoding, energy_encoding, noise_encoding\
             = self.style_encoder(text, speaker_embed, mel_target, p_norm, e_input, mel_aug, mel_len, src_len, src_mask)
        max_seq_len = text_encoding.size(1)
        
        # DAT
        aug_posterior_d = self.augmentation_classifier_d(duration_encoding)
        aug_posterior_p = self.augmentation_classifier_p(pitch_encoding)
        aug_posterior_e = self.augmentation_classifier_e(energy_encoding)

        # Upsampling along the frame domain
        speaker_encoding = speaker_encoding.unsqueeze(1).repeat(1, max_seq_len, 1)
        speaker_encoding_p = speaker_encoding_p.unsqueeze(1).repeat(1, max_seq_len, 1)

        # For the inspection
        self.max_seq_len = max_seq_len
        self.pitch_encoding = pitch_encoding # this should be upsampled before using it during the inspection
        self.speaker_encoding = speaker_encoding
        self.speaker_encoding_p = speaker_encoding_p
        pitch_encoding = pitch_encoding + speaker_encoding_p

        # Upsampling along the channel
        duration_encoding = self.duration_linear(duration_encoding)
        pitch_encoding = self.pitch_linear(pitch_encoding)
        energy_encoding = self.energy_linear(energy_encoding)
        noise_encoding = self.residual_linear(noise_encoding)[:,:max_seq_len]
        text_encoding_neck = self.text_linear_up(text_encoding_neck)

        # For the inspection
        self.text_encoding_neck = text_encoding_neck
        self.duration_encoding = duration_encoding
        self.energy_encoding = energy_encoding
        self.noise_encoding = noise_encoding
        self.text_encoding = text_encoding
        self.src_mask = src_mask
        self.max_len = max_len

        encodings = torch.cat((text_encoding, text_encoding_neck+pitch_encoding, speaker_encoding, text_encoding_neck+energy_encoding, noise_encoding), dim=-1)

        # Duration
        log_duration_prediction = self.duration_predictor(text_encoding_neck + duration_encoding, src_mask) # [batch_size, src_len]
        if duration_target is not None:
            encodings, mel_len = self.length_regulator(encodings, duration_target, max_len)
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction)-hp.log_offset)*d_control), min=0)
            encodings, mel_len = self.length_regulator(encodings, duration_rounded, max_len)
            mel_mask = utils.get_mask_from_lengths(mel_len)

        text_encoding, pitch_encoding, speaker_encoding, energy_encoding, noise_encoding = torch.split(encodings, hp.encoder_hidden, dim=-1)

        # Energy
        energy_prediction = self.energy_predictor(energy_encoding, mel_mask)
        if energy_target is not None:
            energy_embedding = self.energy_embedding(
                torch.bucketize(energy_target, self.energy_bins))
        else:
            energy_prediction = energy_prediction*e_control
            energy_embedding = self.energy_embedding(
                torch.bucketize(energy_prediction, self.energy_bins))

        # Pitch
        pitch_prediction = self.pitch_predictor(pitch_encoding + speaker_encoding, mel_mask)
        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding(
                torch.bucketize(pitch_target, self.pitch_bins))
        else:
            pitch_prediction = pitch_prediction*p_control
            pitch_embedding = self.pitch_embedding(
                torch.bucketize(pitch_prediction, self.pitch_bins))

        # Output
        encoder_output = text_encoding + pitch_embedding + speaker_encoding + energy_embedding # [batch_size, mel_len, encoder_hidden]

        return encoder_output, noise_encoding, log_duration_prediction, pitch_prediction, energy_prediction, mel_len, mel_mask, (aug_posterior_d, aug_posterior_p, aug_posterior_e)


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = utils.pad(output, max_len)
        else:
            output = utils.pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class StylePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self):
        super(StylePredictor, self).__init__()

        self.input_size = hp.encoder_hidden
        self.filter_size = hp.style_predictor_filter_size
        self.kernel = hp.style_predictor_kernel_size
        self.conv_output_size = hp.style_predictor_filter_size
        self.dropout = hp.style_predictor_dropout

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(self.input_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=(self.kernel-1)//2)),
            ("relu_1", nn.ReLU()),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", Conv(self.filter_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("relu_2", nn.ReLU()),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
