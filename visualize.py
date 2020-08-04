import argparse
import torch
from datasets import data_loader
from models.model_utils import model_selector, multipathway_input
from config.m_parser import load_config, arg_parser
from torchviz import make_dot
from graphviz import Source

args = arg_parser().parse_args()
cfg = load_config(args)
plot_name = input('Please specify a name for the plot: ')

x = torch.zeros(1, 3, 32, cfg.DATA.SAMPLE_SIZE, cfg.DATA.SAMPLE_SIZE, dtype=torch.float, requires_grad=False)
model=model_selector(cfg)

if cfg.MODEL.ARCH == 'slowfast':
    x = multipathway_input(x, cfg)

out = model(x)
model_arch_dot = make_dot(out, params=dict(model.named_parameters()))

out_format = 'svg'
model_arch_dot.format = out_format
model_arch_dot.render(plot_name)
print ('Saved graph visualization to {}.{}'.format(plot_name, out_format))
