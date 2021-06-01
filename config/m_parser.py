import argparse
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from default_params import get_cfg


# Argument parser
def arg_parser():
    parser = argparse.ArgumentParser("Video Similarity Search Training Script")
    parser.add_argument(
        "--start_epoch",
        default=None,
        type=int,
        help='overwrite start epoch'
    )
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
        "--cfg", '-cfg',
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
        "--num_data_workers",
        default=None,
        type=int,
        help='define num_workers for dataloader'
    )
    parser.add_argument(
        "--sample_size",
        default=None,
        type=int
    )
    parser.add_argument(
        "--n_classes",
        default=None,
        type=int
    )
    parser.add_argument(
        "--shard_id",
        default=0,
        type=int
    )
    parser.add_argument(
        "--num_shards",
        default=1,
        type=int
    )
    parser.add_argument(
        "--ip_address_port", '-ip',
        default="tcp://localhost:9999",
        type=str
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="See config/defaults.py for all options",
    )
    parser.add_argument(
        '--compute_canada',
        '-cc',
        action='store_true',
        help='Run training with compute canada environment setup'
    )
    parser.add_argument(
        "--sampling_strategy",
        default=None,
        type=str,
        help='Triplet sampling strategy'
    )
    parser.add_argument(
        "--val_metric",
        default=None,
        type=str,
        help='global, local_batch ...'
    )
    parser.add_argument(
        '--val_batch_size',
        default=None,
        type=str,
        help='validation batch size'
    )
    parser.add_argument(
        '--iterative_cluster', '-ic',
        action='store_true',
        help='Perform iterative clustering for pseudolabel assignment'
    )
    parser.add_argument(
        '--vector',
        action='store_true',
        help="running on vector cluster"
    )
    return parser


# Get default cfg and merge parameters from cfg file and opts in arguments
def overwrite_default_configs(cfg, args):
    if args.batch_size:
        cfg.TRAIN.BATCH_SIZE = args.batch_size

    if args.epoch:
        cfg.TRAIN.EPOCHS = args.epoch

    if args.output:
        cfg.OUTPUT_PATH = args.output
        
    if args.num_data_workers:
        cfg.TRAIN.NUM_DATA_WORKERS = args.num_data_workers

    if args.sample_size:
        cfg.DATA.SAMPLE_SIZE = args.sample_size

    if args.sampling_strategy:
        cfg.DATASET.SAMPLING_STRATEGY = args.sampling_strategy

    if args.val_metric:
        cfg.VAL.METRIC = args.val_metric
    if args.val_batch_size:
        cfg.VAL.BATCH_SIZE = int(args.val_batch_size)
    if args.n_classes:
        if cfg.MODEL.ARCH == '3dresnet':
            cfg.RESNET.N_CLASSES = args.n_classes
        else:
            print('not implemented...')


# Return cfg with parameters
def load_config(args):
    cfg = get_cfg()

    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    overwrite_default_configs(cfg, args)

    return cfg
