# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import os
import sys
import json

import torch

from src.utils import Print


class ModelConfig():
    def __init__(self, file=None, idx="model_config"):
        """ model configurations """
        self.idx = idx
        self.num_layers = 40
        self.widening_factor = 2
        self.dropout_rate = None
        self.num_channels = None
        self.num_classes = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("model-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if   key == "num_layers":                   self.num_layers = value
                elif key == "widening_factor":              self.widening_factor = value
                else: sys.exit("# ERROR: invalid key [%s] in model-config file" % key)

    def set_dropout_rate(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def set_num_channels_classes(self, num_channels, num_classes):
        self.num_channels = num_channels
        self.num_classes = num_classes

    def get_config(self):
        configs = []
        configs.append(["num_layers", self.num_layers])
        configs.append(["widening_factor", self.widening_factor])

        return configs


class RunConfig():
    def __init__(self, file=None, idx="run_config", eval=False, sanity_check=False):
        """ model configurations """
        self.idx = idx
        self.eval = eval
        self.batch_size_train = 128
        self.batch_size_eval = 256
        self.num_epochs = 200
        self.learning_rate = 0.1
        self.learning_rate_scheduler = "cyclicLR"
        self.learning_rate_steps = [60, 120, 160]
        self.learning_rate_decay = 0.2
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.dropout_rate = 0.3

        # for adversarial
        self.lower_limit = None
        self.upper_limit = None

        # for adversarial trainijng
        self.adv_train_type = None
        self.epsilon_train = 8
        self.alpha_train = 10
        self.replays_train = 1
        self.restarts_train = 1
        self.attack_iters_train = 1

        # for adversarial evaluation
        self.epsilon_eval = 8
        self.alpha_eval = 2
        self.restarts_eval = 1
        self.attack_iters_eval = 5

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("model-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if   key == "batch_size_train":             self.batch_size_train = value
                elif key == "batch_size_eval":              self.batch_size_eval = value
                elif key == "num_epochs":                   self.num_epochs = value
                elif key == "learning_rate":                self.learning_rate = value
                elif key == "learning_rate_scheduler":
                    if value in ["cyclicLR", "stepLR"]:     self.learning_rate_scheduler = value
                    else: sys.exit("# ERROR: invalid value [%s] for key [%s] in run-config file" % (value, key))
                elif key == "learning_rate_steps":          self.learning_rate_steps = value
                elif key == "learning_rate_decay":          self.learning_rate_decay = value
                elif key == "momentum":                     self.momentum = value
                elif key == "weight_decay":                 self.weight_decay = value
                elif key == "dropout_rate":                 self.dropout_rate = value
                elif key == "adv_train_type":
                    if value in ["fgsm", "free", "pgd"]:    self.adv_train_type = value
                    else: sys.exit("# ERROR: invalid value [%s] for key [%s] in run-config file" % (value, key))
                elif key == "epsilon_train":                self.epsilon_train = value
                elif key == "alpha_train":                  self.alpha_train = value
                elif key == "replays_train":                self.replays_train = value
                elif key == "restarts_train":               self.restarts_train = value
                elif key == "attack_iters_train":           self.attack_iters_train = value
                elif key == "epsilon_eval":                 self.epsilon_eval = value
                elif key == "alpha_eval":                   self.alpha_eval = value
                elif key == "restarts_eval":                self.restarts_eval = value
                elif key == "attack_iters_eval":            self.attack_iters_eval = value
                else: sys.exit("# ERROR: invalid key [%s] in run-config file" % key)

        if sanity_check:
            self.batch_size_train = 32
            self.num_epochs = 4
            if self.replays_train > 1: self.replays_train = 2

    def set_adv(self, info, device):
        mean = torch.tensor(info["mean"]).view(3,1,1).to(device)
        std  = torch.tensor(info["std" ]).view(3,1,1).to(device)
        self.lower_limit = (0 - mean) / std
        self.upper_limit = (1 - mean) / std
        self.epsilon_train = (self.epsilon_train / 255.0) / std
        self.alpha_train   = (self.alpha_train   / 255.0) / std
        self.epsilon_eval  = (self.epsilon_eval  / 255.0) / std
        self.alpha_eval    = (self.alpha_eval    / 255.0) / std

    def get_config(self):
        configs = []
        if not self.eval:
            configs.append(["batch_size_train", self.batch_size_train])
        configs.append(["batch_size_eval", self.batch_size_eval])
        if not self.eval:
            configs.append(["num_epochs", self.num_epochs])
            configs.append(["learning_rate", self.learning_rate])
            configs.append(["learning_rate_scheduler", self.learning_rate_scheduler])
            if self.learning_rate_scheduler == "stepLR":
                configs.append(["learning_rate_steps", self.learning_rate_steps])
                configs.append(["learning_rate_decay", self.learning_rate_decay])
            configs.append(["momentum", self.momentum])
            configs.append(["weight_decay", self.weight_decay])
            configs.append(["dropout_rate", self.dropout_rate])
            if self.adv_train_type is not None:
                configs.append(["adv_train_type", self.adv_train_type])
                if self.adv_train_type == "fgsm":
                    configs.append(["epsilon_train", self.epsilon_train])
                    configs.append(["alpha_train", self.alpha_train])
                elif self.adv_train_type == "free":
                    configs.append(["epsilon_train", self.epsilon_train])
                    configs.append(["replays_train", self.replays_train])
                elif self.adv_train_type == "pgd":
                    configs.append(["epsilon_train", self.epsilon_train])
                    configs.append(["alpha_train", self.alpha_train])
                    configs.append(["restarts_train", self.restarts_train])
                    configs.append(["attack_iters_train", self.attack_iters_train])
                configs.append(["epsilon_eval", self.epsilon_eval])
                configs.append(["alpha_eval", self.alpha_eval])
                configs.append(["restarts_eval", self.restarts_eval])
                configs.append(["attack_iters_eval", self.attack_iters_eval])

        return configs


def print_configs(args, cfgs, device, output):
    if args["sanity_check"]: Print(" ".join(['##### SANITY_CHECK #####']), output)
    Print(" ".join(['##### arguments #####']), output)
    Print(" ".join(['dataset:', args["dataset"]]), output)
    for cfg in cfgs:
        Print(" ".join(['%s:' % cfg.idx, str(args[cfg.idx])]), output)
        for c, v in cfg.get_config():
            Print(" ".join(['-- %s: %s' % (c, v)]), output)
    if args["checkpoint"] is not None:
        Print(" ".join(['checkpoint: %s' % (args["checkpoint"])]), output)
    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)
    Print(" ".join(['output_path:', str(args["output_path"])]), output)
    Print(" ".join(['log_file:', str(output.name)]), output, newline=True)