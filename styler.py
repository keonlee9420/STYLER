import torch
import torch.nn as nn

from transformer.Models import Decoder
from transformer.Layers import PostNet
from modules import StyleModeling
from utils import get_mask_from_lengths
import hparams as hp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class STYLER(nn.Module):
    """ STYLER """

    def __init__(self, use_postnet=True):
        super(STYLER, self).__init__()

        self.style_modeling = StyleModeling()

        self.decoder = Decoder()
        self.mel_linear = nn.Linear(hp.decoder_hidden, hp.n_mel_channels)

        self.use_postnet = use_postnet
        if self.use_postnet:
            self.postnet = PostNet()
        encoder_output = None

    def decode(self, style_modeling_output, mel_mask):
        decoder_output = self.decoder(style_modeling_output, mel_mask)
        mel_output = self.mel_linear(decoder_output)

        if self.use_postnet:
            mel_output_postnet = self.postnet(mel_output) + mel_output
        else:
            mel_output_postnet = mel_output
        return mel_output, mel_output_postnet

    def forward(self, src_seq, mel_target, mel_aug, p_norm, e_input, src_len, mel_len, d_target=None, p_target=None, e_target=None, max_src_len=None, max_mel_len=None, speaker_embed=None, d_control=1.0, p_control=1.0, e_control=1.0):
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len)

        # Style modeling
        if d_target is not None:
            style_modeling_output, noise_encoding, d_prediction, p_prediction, e_prediction, _, _, (aug_posterior_d, aug_posterior_p, aug_posterior_e) = self.style_modeling(
                src_seq, speaker_embed, mel_target, mel_aug, p_norm, e_input, src_len, mel_len, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len, d_control, p_control, e_control)
        else:
            style_modeling_output, noise_encoding, d_prediction, p_prediction, e_prediction, mel_len, mel_mask, (aug_posterior_d, aug_posterior_p, aug_posterior_e) = self.style_modeling(
                src_seq, speaker_embed, mel_target, mel_aug, p_norm, e_input, src_len, mel_len, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len, d_control, p_control, e_control)

        # Clean decoding
        mel_output, mel_output_postnet = self.decode(style_modeling_output, mel_mask)

        # Noisy decoding
        mel_output_noisy, mel_output_postnet_noisy = self.decode(style_modeling_output.detach() + noise_encoding, mel_mask)

        return (mel_output, mel_output_noisy), (mel_output_postnet, mel_output_postnet_noisy), d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len, \
            (aug_posterior_d, aug_posterior_p, aug_posterior_e)