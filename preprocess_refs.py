import argparse
import os
from data import references
import hparams as hp
import utils


def main(args):
    in_dir = args.data_dir
    out_dir = os.path.join(hp.preprocessed_basedir, args.data_dir.split("/")[-1])

    # Setup directories
    mel_out_dir = os.path.join(out_dir, "mel")
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

    # Prepare align
    references.prepare_align(in_dir)

    # MFA
    mfa_out_dir = utils.mfa(in_dir, out_dir)

    # Build preprocessed dataset
    basenames, audio_paths = references.build_from_path(in_dir, out_dir)

    # Save wav path and TextGrid path pair
    with open(os.path.join(out_dir, 'wav_tg_pairs.txt'), 'w', encoding='utf-8') as f:
        for audio_path in audio_paths:
            basename = audio_path.split('/')[-1].replace('.wav', '')
            if basename in basenames:
                tg_path = os.path.join(mfa_out_dir, basename+'.TextGrid')
                row = '|'.join([audio_path, tg_path])
                f.write(row+'\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=hp.ref_audio_dir)
    args = parser.parse_args()

    main(args)
