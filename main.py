from trainer import Trainer
from inference import BeamSearcher
import config
import argparse


def main(args):
    if args.train:
        trainer = Trainer(args)
        trainer.train()
    else:
        beamsearcher = BeamSearcher(args.model_path, args.output_dir)
        beamsearcher.decode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=True ,action="store_true")
    parser.add_argument("--model_path", type=str, default="",
                        help="path to the saved checkpoint")
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    main(args)
