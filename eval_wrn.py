# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import os
import sys
import argparse

import torch
import torch.nn as nn

import src.config as config
from src.data import get_dataset
from src.model import WideResNet
from src.train import Trainer
from src.utils import Print, set_seeds, set_output, check_args


parser = argparse.ArgumentParser('Evaluating a WideResNet Model')
parser.add_argument('--dataset',  help='path for data configuration file')
parser.add_argument('--model-config', help='path for model configuration file')
parser.add_argument('--run-config', help='path for run configuration file')
parser.add_argument('--checkpoint', help='path for checkpoint to resume')
parser.add_argument('--device', help='device to use; multi-GPU if given multiple GPUs sperated by comma (default: cpu)')
parser.add_argument('--output-path', help='path for outputs (default: stdout and without saving)')
parser.add_argument('--sanity-check', default=False, action='store_true', help='sanity check flag')


def main():
    args = vars(parser.parse_args())
    check_args(args)
    set_seeds(2020)
    model_cfg = config.ModelConfig(args["model_config"])
    run_cfg   = config.RunConfig(args["run_config"], eval=True, sanity_check=args["sanity_check"])
    output, writer, save_prefix = set_output(args, "eval_wrn_log")
    os.environ['CUDA_VISIBLE_DEVICES'] = args["device"] if args["device"] is not None else ""
    device, data_parallel = torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.cuda.device_count() > 1
    config.print_configs(args, [model_cfg, run_cfg], device, output)

    ## Loading datasets
    start = Print(" ".join(['start loading datasets:', args["dataset"]]), output)
    dataset_test , dataset_info  = get_dataset(args["dataset"], test=True,  sanity_check=args["sanity_check"])
    iterator_test  = torch.utils.data.DataLoader(dataset_test,  run_cfg.batch_size_eval,  shuffle=True, num_workers=2)
    end = Print(" ".join(['loaded', str(len(dataset_test )), 'dataset_test samples']), output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## initialize a model
    start = Print('start initializing a model', output)
    model_cfg.set_num_channels_classes(dataset_info["num_channels"], dataset_info["num_classes"])
    model_cfg.set_dropout_rate(run_cfg.dropout_rate)
    model = WideResNet(model_cfg)
    end = Print('end initializing a model', output)
    Print("".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## setup trainer configurations
    start = Print('start setting trainer configurations', output)
    if not data_parallel: model = model.to(device)
    else:                 model = nn.DataParallel(model.to(device))
    criterion = nn.CrossEntropyLoss(reduction="none")
    run_cfg.set_adv(dataset_info, device)
    trainer = Trainer(model, criterion, run_cfg, std=True, adv=True, test=True)
    trainer.load(args["checkpoint"], save_prefix, device, output)
    end = Print('end setting trainer configurations', output)
    Print("".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## train a model
    start = Print('start evaluating a model', output)
    Print(trainer.get_headline(), output)
    ### test
    for B, batch in enumerate(iterator_test):
        batch = [t.to(device) if type(t) is torch.Tensor else t for t in batch]
        trainer.std_evaluate(batch)
        trainer.adv_evaluate(batch)
        if B % 2 == 0: print('# test {:.1%}'.format(B / len(iterator_test)), end='\r', file=sys.stderr)
    print(' ' * 150, end='\r', file=sys.stderr)

    ### print log and save models
    trainer.log(output, writer)

    end = Print('end evaluating a model', output)
    Print("".join(['elapsed time:', str(end - start)]), output, newline=True)
    if not output == sys.stdout: output.close()

if __name__ == '__main__':
    main()
