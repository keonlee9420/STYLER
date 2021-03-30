import argparse
import os
from data import noise_mixer, noise_mixer_refs
import hparams as hp


def main(args):
    in_dir = hp.data_dir
    out_dir = hp.preprocessed_path

    if args.refs:
        noise_mixer_refs.build_from_path(hp.ref_audio_dir)
        return

    mel_aug_out_dir = os.path.join(out_dir, "mel_aug")
    if not os.path.exists(mel_aug_out_dir):
        os.makedirs(mel_aug_out_dir, exist_ok=True)
    
    if hp.dataset == "VCTK":
        noise_mixer.build_from_path(in_dir, out_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--refs', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
