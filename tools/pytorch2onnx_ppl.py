import argparse
from functools import partial

import mmcv
import numpy as np
import onnxruntime as rt
import torch
import torch._C
import torch.serialization
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint
from torch import nn

from mmseg.models import build_segmentor

import warnings
import requests
import os
import argparse
import warnings
import os.path as osp
import onnxtool
torch.manual_seed(3)


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


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
    segs = rng.randint(
        low=0, high=num_classes - 1, size=(N, 1, H, W)).astype(np.uint8)
    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
    } for _ in range(N)]
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_semantic_seg': torch.LongTensor(segs)
    }
    return mm_inputs


def pytorch2onnx(model,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx'):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """
    model.cpu().eval()

    if isinstance(model.decode_head, nn.ModuleList):
        num_classes = model.decode_head[-1].num_classes
    else:
        num_classes = model.decode_head.num_classes

    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    img_list = [img[None, :] for img in imgs]
    print(img_list)
    img_meta_list = [[img_meta] for img_meta in img_metas]

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(
        model.forward, img_metas=img_meta_list, return_loss=False)

    register_extra_symbolics(opset_version)
    with torch.no_grad():
        input_tensors = { 'input_mms': img_list }
        output_names = ['output_mms']

        output_tensors = onnxtool.ppl_export(model,
                  input_tensors,
                  output_file,
                  output_names,
                  opset_version)
        """
        
        torch.onnx.export(
            model, (img_list, ),
            output_file,
            export_params=True,
            keep_initializers_as_inputs=True,
            verbose=show,
            opset_version=opset_version)
        """
        print(f'Successfully exported ONNX model: {output_file}')
    model.forward = origin_forward

    pytorch_result = model(img_list, img_meta_list, return_loss=False)[0]
    print(pytorch_result)
    print(output_tensors['output_mms'])

    if not np.allclose(pytorch_result, output_tensors['output_mms'][0]):
        raise ValueError(
            'The outputs are different between Pytorch and ONNX')
    print('The outputs are same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMSeg to ONNX')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./checkpoints',
        help='checkpoint directory')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument(
        '--verify', action='store_true', help='verify the onnx model')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[256, 256],
        help='input image size')
    parser.add_argument('--all', action='store_true',
                        help='run ci for all the cases in ../configs  ')

    args = parser.parse_args()
    return args



def list_all_files(rootdir):
    import os
    _files = []
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           #print ('path is',path)

           if os.path.isdir(path) :
               _files.extend(list_all_files(path))

           if os.path.isfile(path):
              _files.append(path)
    return _files

def download_from_url(url, path):
    print(f'Downloading: {url}')
    resp = requests.get(url)
    success = True
    if resp.status_code != 200:
        warnings.warn(f'Failed to download {url}')
        success = False
    out_dir, _ = osp.split(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(resp.content)
    return success


def parse_urls(readme_path):
    url_li = []
    domain = 'http'
    start = '[model]('

    f = open(readme_path, 'r', encoding='UTF-8')

    #with open(readme_path, 'r') as f:
        #print(f)
    for line in f:
        if domain in line and start in line:
            url = line.split(start)[1].split('.pth')[0] + '.pth'
            url_li.append(url)
    return url_li


if __name__ == '__main__':
    args = parse_args()

    print('arg done')
    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    print('shape done')
    #download pth
    #if args.checkpoint:
    #    load_checkpoint(segmentor, args.checkpoint, map_location='cpu')

    files = list_all_files("../configs")

    testcases = []
    for cases in files:
        flag_base = 0
        for folders in cases.split('/'):
            # print(folders)
            if folders == '_base_':
                flag_base = 1
        if flag_base == 0:
            if cases.split('.')[-1] == 'py':
                testcases.append(cases)
    print('cases done')

    # print(testcases)
    results = []
    header = [
        'algo_name', 'model_name', 'status', 'exported_onnx', 'error',
        'message'
    ]
    i = 0
    test_folders = ['fcn', 'pspnet', 'deeplabv3', 'deeplabv3plus']
    for cases in testcases:
        if cases.split('/')[2] not in test_folders:
            continue

        checkpoint_dir = args.checkpoint
        config_path = cases

        cfg = mmcv.Config.fromfile(config_path)
        cfg.model.pretrained = None

        # build the model and load checkpoint
        segmentor = build_segmentor(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        # convert SyncBN to BN
        segmentor = _convert_batchnorm(segmentor)


        config_dir, fname = osp.split(cases)
        _, algo_name = osp.split(config_dir)
        key, _ = osp.splitext(fname)
        parent_dir = osp.join(checkpoint_dir, algo_name)
        os.makedirs(parent_dir, exist_ok=True)
        checkpoint_path = osp.join(parent_dir, key + '.pth')

        #print('checkpoint_path ', checkpoint_path,parent_dir, key)
        if not osp.exists(checkpoint_path):
            readme_path = osp.join(config_dir, 'README.md')
            if not osp.exists(readme_path):
                warnings.warn(f'No README.md found in {config_dir}, \
                        could not get checkpoint for {config_path}')
                continue
            url_li = parse_urls(readme_path)
            for url_liss in url_li:
                print('url_liss ', url_liss)

            #reverse
            url_li.reverse()
            url_maps = {u.split('/')[-2]: u for u in url_li}
            for urlss in url_maps:
                print('urlss ', urlss)
            url = None
            print('key ', key)
            if key in url_maps:
                url = url_maps[key]
            else:
                for u in url_li:
                    if key in u:
                        url = u
                        break
            if url is None:
                warnings.warn(f'Failed to get checkpoint url for {config_path}')
                continue
            print('url ', url,checkpoint_path )
            success = download_from_url(url, checkpoint_path)
            if not success:
                continue


        # conver model to onnx file
        load_checkpoint(segmentor, checkpoint_path, map_location='cpu')


        pytorch2onnx(
                segmentor,
                input_shape,
                opset_version=args.opset_version,
                show=args.show,
                output_file=args.output_file)
