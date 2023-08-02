import argparse
from configs import cfg
import os
from src.tracker import Tracker

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID


if __name__ == "__main__":
    parse_args()
    trainer = Tracker(cfg)
    if cfg.MODEL.MODE == 'train':
        trainer.train()
    elif cfg.MODEL.MODE == 'test':
        trainer.test()
    else:
        raise ValueError("Please assign a state (train, test)")