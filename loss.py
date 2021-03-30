import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class STYLERLoss(nn.Module):
    """ STYLER Loss """

    def __init__(self):
        super(STYLERLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.nll_loss = nn.NLLLoss()

    def cal_mel_loss(self, mel, mel_postnet, mel_target, mel_mask):
        mel_target.requires_grad = False
        mel = mel.masked_select(mel_mask.unsqueeze(-1))
        mel_postnet = mel_postnet.masked_select(mel_mask.unsqueeze(-1))
        mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))

        mel_loss = self.mse_loss(mel, mel_target)
        mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)
        return mel_loss, mel_postnet_loss

    def forward(self, log_d_predicted, log_d_target, p_predicted, p_target, e_predicted, e_target, mel, mel_postnet, mel_target, src_mask, mel_mask, src_len, mel_len, aug_posteriors, aug_label):
        log_d_target.requires_grad = False
        p_target.requires_grad = False
        e_target.requires_grad = False
        aug_label.requires_grad = False

        aug_posterior_d, aug_posterior_p, aug_posterior_e = aug_posteriors
        log_d_predicted = log_d_predicted.masked_select(src_mask)
        log_d_target = log_d_target.masked_select(src_mask)
        p_predicted = p_predicted.masked_select(mel_mask)
        p_target = p_target.masked_select(mel_mask)
        e_predicted = e_predicted.masked_select(mel_mask)
        e_target = e_target.masked_select(mel_mask)

        mel_loss, mel_postnet_loss = self.cal_mel_loss(mel, mel_postnet, mel_target, mel_mask)

        d_loss = self.mae_loss(log_d_predicted, log_d_target)
        p_loss = self.mae_loss(p_predicted, p_target)
        e_loss = self.mae_loss(e_predicted, e_target)

        classifier_loss_a = self.nll_loss(aug_posterior_d, aug_label)
        classifier_loss_a += self.nll_loss(aug_posterior_p, aug_label)
        classifier_loss_a += self.nll_loss(aug_posterior_e, aug_label)

        return mel_loss, mel_postnet_loss, d_loss, p_loss, e_loss, classifier_loss_a


class DomainAdversarialTrainingLoss(nn.Module):
    """ Domain Adversarial Training Loss """

    def __init__(self):
        super(DomainAdversarialTrainingLoss, self).__init__()
        self.nll_loss = nn.NLLLoss()

    def forward(self, augmentation_posterior, aug_label):
        aug_posterior_d, aug_posterior_p, aug_posterior_e = augmentation_posterior
        aug_label.requires_grad = False

        # Domain adaptation loss for target(augmented noisy data)
        classifier_loss_a = self.nll_loss(aug_posterior_d, aug_label)
        classifier_loss_a += self.nll_loss(aug_posterior_p, aug_label)
        classifier_loss_a += self.nll_loss(aug_posterior_e, aug_label)
        return classifier_loss_a