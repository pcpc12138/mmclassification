import argparse
from functools import partial
import json

import mmcv
import numpy as np
import torch
if torch.__version__ != 'parrots':
    raise ImportError('Please use parrots not pytorch')
from parrotsconvert.caffe import CaffeNet, BackendSet
from mmcv.runner import load_checkpoint

from mmcls.models import build_classifier

torch.manual_seed(3)


def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    gt_labels = rng.randint(
        low=0, high=num_classes - 1, size=(N, 1)).astype(np.uint8)
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'gt_labels': torch.LongTensor(gt_labels),
    }
    return mm_inputs


def parrots2ppl(model,
                input_shape,
                output_file='./tmp'):
    model.cpu().eval()

    num_classes = model.head.num_classes
    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop('imgs')
    img_list = [img[None, :] for img in imgs]

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(model.forward, return_loss=False)

    with torch.no_grad():
        input = img_list
        caffe_net = CaffeNet(model, input, BackendSet.PPL)
        caffe_net.dump_model(output_file)
    model.forward = origin_forward



def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMCls to ppl-caffe')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--output-file', type=str, default='./tmp')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)

    # dump model info
    with open('info.json', 'w') as f:
        f.write(json.dumps(cfg._cfg_dict))
    cfg.model.pretrained = None

    # build the model and load checkpoint
    classifier = build_classifier(cfg.model)

    if args.checkpoint:
        load_checkpoint(classifier, args.checkpoint, map_location='cpu')

    # conver model to ppl-caffe file
    parrots2ppl(
        classifier,
        input_shape,
        output_file=args.output_file)
