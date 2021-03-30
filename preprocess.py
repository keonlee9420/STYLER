import os
from data import vctk
import hparams as hp
import utils


def write_metadata(train, val, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in train:
            f.write(m + '\n')
    with open(os.path.join(out_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        for m in val:
            f.write(m + '\n')


def main():
    in_dir = hp.data_dir
    out_dir = hp.preprocessed_path
    mel_out_dir = os.path.join(out_dir, "mel_clean")
    if not os.path.exists(mel_out_dir):
        os.makedirs(mel_out_dir, exist_ok=True)
    ali_out_dir = os.path.join(out_dir, "alignment")
    if not os.path.exists(ali_out_dir):
        os.makedirs(ali_out_dir, exist_ok=True)
    f0_out_dir = os.path.join(out_dir, "f0")
    if not os.path.exists(f0_out_dir):
        os.makedirs(f0_out_dir, exist_ok=True)
    f0_norm_out_dir = os.path.join(out_dir, "f0_norm")
    if not os.path.exists(f0_norm_out_dir):
        os.makedirs(f0_norm_out_dir, exist_ok=True)
    energy_out_dir = os.path.join(out_dir, "energy")
    if not os.path.exists(energy_out_dir):
        os.makedirs(energy_out_dir, exist_ok=True)
    energy_0to1_out_dir = os.path.join(out_dir, "energy_0to1")
    if not os.path.exists(energy_0to1_out_dir):
        os.makedirs(energy_0to1_out_dir, exist_ok=True)
    spker_out_dir = os.path.join(out_dir, "spker_embed")
    if not os.path.exists(spker_out_dir):
        os.makedirs(spker_out_dir, exist_ok=True)

    if hp.dataset == "VCTK":
        # Prepare align
        vctk.prepare_align(in_dir)

        # MFA
        mfa_out_dir = utils.mfa(in_dir, out_dir)

        # Build preprocessed dataset
        train, val = vctk.build_from_path(in_dir, out_dir)
    
    # Write Filelist
    write_metadata(train, val, out_dir)


if __name__ == "__main__":
    main()
