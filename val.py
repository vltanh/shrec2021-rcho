import argparse

import yaml
import torch
from tqdm import tqdm

from torchan.utils.getter import get_instance, get_single_data
from torchan.utils.device import move_to, detach


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        type=str,
                        help='path to configuration file')
    parser.add_argument('-g', '--gpu',
                        type=int,
                        default=None,
                        help='(single) GPU to use (default: None)')
    return parser.parse_args()


@torch.no_grad()
def val(model, dataloader, metrics, device):
    model.eval()
    print('Evaluating........')
    progress_bar = tqdm(dataloader)
    for i, (inp, lbl) in enumerate(progress_bar):
        # Load inputs and labels
        inp = move_to(inp, device)
        lbl = move_to(lbl, device)

        # Get network outputs
        outs = model(inp)

        # Update metrics
        outs = detach(outs)
        lbl = detach(lbl)
        for _, m in metrics:
            m.update(outs, lbl)

    print('== Evaluation result ==')
    for _, m in metrics:
        m.summary()


def generate_device(gpu):
    dev_id = 'cuda:{}'.format(gpu) \
        if torch.cuda.is_available() and gpu is not None \
        else 'cpu'
    device = torch.device(dev_id)
    return dev_id, device


def generate_model(pretrained, dev_id, device):
    model_cfg = torch.load(pretrained, map_location=dev_id)
    model = get_instance(model_cfg['config']['model']).to(device)
    model.load_state_dict(model_cfg['model_state_dict'])
    return model


args = parse_args()

# Load val config
config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

# Device
dev_id, device = generate_device(args.gpu)

# Load model
model = generate_model(config['pretrained'], dev_id, device)

# Load data
dataloader = get_single_data(config['dataset'], with_dataset=False)

# Define metrics
metrics = [
    (mcfg['name'], get_instance(mcfg))
    for mcfg in config['metric']
]

# Perform validation and print result
val(model, dataloader, metrics, device)
