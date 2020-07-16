import argparse
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from default_params import get_cfg


# Argument parser
def parse_args():
    parser = argparse.ArgumentParser("Video Similarity Search Training Script")
    parser.add_argument(
        '--pretrain_path',
        default=None,
        type=str, action='store',
        help='Path to pretrained encoder'
    )
    parser.add_argument(
        '--checkpoint_path',
        default=None,
        type=str, action='store',
        help='Path to checkpoint'
    )
    parser.add_argument(
        "--cfg",
        default=None,
        dest="cfg_file", type=str,
        help="Path to the config file",
    )
    parser.add_argument(
        '--gpu',
        default='0,1', type=str
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
    help='output path, overwrite OUTPUT_PATH in default_params.py if specified'
    )
    parser.add_argument(
        "--batch_size",
        default=None,
        type=int,
        help='overwrite batch size'
    )
    parser.add_argument(
        "--epoch",
        default=None,
        type=int,
        help='define number of epoch'
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="See config/defaults.py for all options",
    )
    return parser.parse_args()


# Get default cfg and merge parameters from cfg file and opts in arguments

def overwrite_default_configs(cfg, args):
    if args.batch_size:
        cfg.TRAIN.BATCH_SIZE = args.batch_size

    if args.epoch:
        cfg.TRAIN.EPOCHS = args.epoch

    if args.output:
        cfg.OUTPUT_PATH = args.output

def load_config(args):
    cfg = get_cfg()

    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    overwrite_default_configs(cfg, args)

    print('\nOUTPUT_PATH is set to: {}'.format(cfg.OUTPUT_PATH))
    print('BATCH_SIZE is set to: {}'.format(cfg.TRAIN.BATCH_SIZE))
    print('NUM_WORKERS is set to: {}'.format(cfg.TRAIN.NUM_DATA_WORKERS))

    return cfg
