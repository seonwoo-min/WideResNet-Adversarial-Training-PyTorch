# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

""" Utility functions """

import os
import sys
import random
import numpy as np
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter


def Print(string, output, newline=False):
    """ print to stdout and a file (if given) """
    time = datetime.now()
    print('\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string]), file=sys.stderr)
    if newline: print("", file=sys.stderr)

    if not output == sys.stdout:
        print('\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string]), file=output)
        if newline: print("", file=output)

    output.flush()
    return time


def set_seeds(seed):
    """ set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_args(args):
    """ sanity check for arguments """
    if args["dataset"] not in ["cifar10", "cifar100"]:
        sys.exit("dataset [%s] is not supported" % args["dataset"])
    if args["checkpoint"] is not None and not os.path.exists(args["checkpoint"]):
            sys.exit("checkpoint [%s] does not exists" % (args["checkpoint"]))


def set_output(args, string):
    """ set output configurations """
    output, writer, save_prefix = sys.stdout, None, None
    if args["output_path"] is not None:
        save_prefix = args["output_path"] + "/checkpoints/"
        if not os.path.exists(save_prefix):
            os.makedirs(save_prefix, exist_ok=True)
        output = open(args["output_path"] + "/" + string + ".txt", "a")
        if "eval" not in string:
            tb = args["output_path"] + "/tensorboard/"
            if not os.path.exists(tb):
                os.makedirs(tb, exist_ok=True)
            writer = SummaryWriter(tb)

    return output, writer, save_prefix


def clamp(x, lower_limit, upper_limit):
    """ clamp x to given limits """
    return torch.max(torch.min(x, upper_limit), lower_limit)
