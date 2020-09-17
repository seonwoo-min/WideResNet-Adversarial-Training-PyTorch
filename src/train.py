# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# https://github.com/locuslab/fast_adversarial

import os
from collections import OrderedDict

import torch
import torch.nn as nn

from src.utils import Print, clamp


class Trainer():
    """ train / eval helper class """
    def __init__(self, model, criterion, run_cfg, std=False, adv=False, test=False):
        self.model = model
        self.criterion = criterion
        self.run_cfg = run_cfg
        self.std = std
        self.adv = adv
        self.test = test
        self.optim = None
        self.scheduler = None

        # perturbation parameter for free adversarial training
        self.delta = None

        # initialize logging parameters
        self.epoch = 0.0
        self.logger_std_train = Logger("std")
        self.logger_std_test  = Logger("std")
        self.logger_adv_train = Logger("adv")
        self.logger_adv_test  = Logger("adv")

    def std_train(self, batch):
        # standard training of the model
        self.model.train()
        inputs, labels = batch

        self.optim.zero_grad()
        outputs = self.model(inputs)
        _, predictions = torch.max(outputs, 1)
        loss = torch.mean(self.criterion(outputs, labels))
        loss.backward()
        self.optim.step()
        self.scheduler.step()

        # logging
        self.logger_std_train.update(len(inputs), (predictions == labels).sum().item(), loss.item())

    def std_evaluate(self, batch):
        # standard evaluation of the model
        self.model.eval()
        inputs, labels = batch

        with torch.no_grad():
            outputs = self.model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = torch.mean(self.criterion(outputs, labels))

            # logging
            self.logger_std_test.update(len(inputs), (predictions == labels).sum().item(), loss.item())

    def adv_train(self, batch):
        # adversarial training of the model
        self.model.train()
        inputs, labels = batch

        for replay in range(self.run_cfg.replays_train):
            delta = self.adv_attack(inputs, labels, self.run_cfg.epsilon_train,  self.run_cfg.alpha_train,
                                    self.run_cfg.restarts_train, self.run_cfg.attack_iters_train, eval=False)

            self.optim.zero_grad()
            outputs = self.model(inputs + delta)
            _, predictions = torch.max(outputs, 1)
            loss = torch.mean(self.criterion(outputs, labels))
            loss.backward()
            self.optim.step()
            self.scheduler.step()

        # logging
        self.logger_adv_train.update(len(inputs), (predictions == labels).sum().item(), loss.item())

    def adv_evaluate(self, batch):
        # adversarial evaluation of the model
        self.model.eval()
        inputs, labels = batch
        delta = self.adv_attack(inputs, labels, self.run_cfg.epsilon_eval, self.run_cfg.alpha_eval,
                                self.run_cfg.restarts_eval, self.run_cfg.attack_iters_eval, eval=True)

        with torch.no_grad():
            outputs = self.model(inputs + delta)
            _, predictions = torch.max(outputs, 1)
            loss = torch.mean(self.criterion(outputs, labels))

            # logging
            self.logger_adv_test.update(len(inputs), (predictions == labels).sum().item(), loss.item())

    def adv_attack(self, inputs, labels, epsilon, alpha, restarts, attack_iters, eval=False):
        # adversarial attack to obtain perturbation delta
        lower_limit = self.run_cfg.lower_limit
        upper_limit = self.run_cfg.upper_limit

        max_delta = torch.zeros_like(inputs)
        max_loss  = torch.zeros_like(labels, dtype=torch.float)
        for restart in range(restarts):
            delta = torch.zeros_like(inputs)
            if self.delta is not None and not eval and self.run_cfg.adv_train_type == "free":
                delta.data = self.delta.data
            else:
                for i in range(delta.shape[1]):
                    delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
            delta.data[:len(inputs)] = clamp(delta[:len(inputs)], lower_limit - inputs, upper_limit - inputs)
            delta.requires_grad = True

            for attack_iter in range(attack_iters):
                outputs = self.model(inputs + delta[:len(inputs)])
                if eval: index = torch.where(torch.max(outputs, 1)[1] == labels)[0]
                else:    index = torch.arange(len(inputs)).to(inputs.device)
                if len(index) == 0: break

                loss = torch.mean(self.criterion(outputs, labels))
                loss.backward()
                d = delta[index]
                d = clamp(d + alpha * torch.sign(delta.grad.detach()[index]), -epsilon, epsilon)
                d = clamp(d, lower_limit - inputs[index], upper_limit - inputs[index])
                delta.data[index] = d
                delta.grad.zero_()

            outputs = self.model(inputs + delta[:len(inputs)])
            loss = self.criterion(outputs, labels).detach()
            max_delta[loss >= max_loss] = delta.detach()[:len(inputs)][loss >= max_loss]
            max_loss = torch.max(max_loss, loss)

        if not eval and self.run_cfg.adv_train_type == "free":
            if self.delta is None: self.delta = max_delta
            else:                  self.delta[:len(max_delta)] = max_delta

        return max_delta

    def save(self, save_prefix):
        # save state_dicts to checkpoint """
        if save_prefix is None: return
        state = {}
        state["model"] = self.model.state_dict()
        state["optim"] = self.optim.state_dict()
        state["scheduler"] = self.scheduler.state_dict()
        state["epoch"] = self.epoch
        torch.save(state, save_prefix+"%d.pt" % self.epoch)

    def load(self, checkpoint, save_prefix, device, output):
        # load state_dicts from checkpoint """
        if checkpoint is None:
            if save_prefix is None: return
            checkpoints = [os.path.splitext(file)[0] for file in os.listdir(save_prefix)]
            checkpoints = sorted([int(checkpoint) for checkpoint in checkpoints if not checkpoint.startswith("final")])
            if len(checkpoints) == 0: return
            checkpoint = save_prefix + "%d.pt" % checkpoints[-1]
            Print('resuming from the last checkpoint [%s]' % (checkpoint), output)

        Print('loading a model state_dict from the checkpoint', output)
        checkpoint = torch.load(checkpoint, map_location="cpu")
        state_dict = OrderedDict()
        for k, v in checkpoint["model"].items():
            if k.startswith("module."): k = k[7:]
            state_dict[k] = v
        self.model.load_state_dict(state_dict)
        if self.optim is not None:
            Print('loading a optim state_dict from the checkpoint', output)
            self.optim.load_state_dict(checkpoint["optim"])
            Print('loading a scheduler state_dict from the checkpoint', output)
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            Print('loading current epoch from the checkpoint', output)
            self.epoch = checkpoint["epoch"]

    def set_device(self, device, data_parallel):
        # set gpu configurations
        self.model = self.model.to(device)
        if data_parallel:
            self.model = nn.DataParallel(self.model)

    def set_optim_scheduler(self, params, run_cfg):
        # set optim and scheduler for training
        optim, scheduler = get_optim_scheduler(params, run_cfg)
        self.optim = optim
        self.scheduler = scheduler
        self.run_cfg = run_cfg

    def get_headline(self):
        # get a headline for logging
        headline = []
        if not self.test:
            headline += ["ep", "split"]
            if self.std: headline += self.logger_std_train.get_headline()
            if self.adv: headline += self.logger_adv_train.get_headline()
            headline += ["|"]

        headline += ["split"]
        if self.std: headline += self.logger_std_test.get_headline()
        if self.adv: headline += self.logger_adv_test.get_headline()

        return "\t".join(headline)

    def log(self, output, writer=None):
        # logging
        log, log_dict = [], {}

        if not self.test:
            log += ["%03d" % self.epoch, "train"]
            if self.std:
                self.logger_std_train.evaluate()
                log += self.logger_std_train.log
                if writer is not None:
                    for k, v in self.logger_std_train.log_dict.items():
                        if k not in log_dict: log_dict[k] = {}
                        log_dict[k]["train"] = v
            if self.adv:
                self.logger_adv_train.evaluate()
                log += self.logger_adv_train.log
                if writer is not None:
                    for k, v in self.logger_adv_train.log_dict.items():
                        if k not in log_dict: log_dict[k] = {}
                        log_dict[k]["train"] = v
            log += ["|"]

        log += ["test"]
        if self.std:
            self.logger_std_test.evaluate()
            log += self.logger_std_test.log
            if writer is not None:
                for k, v in self.logger_std_test.log_dict.items():
                    if k not in log_dict: log_dict[k] = {}
                    log_dict[k]["test"] = v
        if self.adv:
            self.logger_adv_test.evaluate()
            log += self.logger_adv_test.log
            if writer is not None:
                for k, v in self.logger_adv_test.log_dict.items():
                    if k not in log_dict: log_dict[k] = {}
                    log_dict[k]["test"] = v

        Print("\t".join(log), output)
        if writer is not None:
            for k, v in log_dict.items():
                writer.add_scalars(k, v, self.epoch)
            writer.flush()
        self.log_reset()

    def log_reset(self):
        # reset logging parameters
        self.logger_std_train.reset()
        self.logger_std_test.reset()
        self.logger_adv_train.reset()
        self.logger_adv_test.reset()


class Logger():
    """ Logger class """
    def __init__(self, idx):
        self.idx = idx
        self.total = 0.0
        self.correct = 0.0
        self.loss = 0.0
        self.log = []
        self.log_dict = {}

    def update(self, total, correct, loss):
        self.total += total
        self.correct += correct
        self.loss += loss * total

    def get_headline(self):
        # get a headline
        headline = [
            "%s_loss" % self.idx,
            "%s_acc"  % self.idx
        ]

        return headline

    def evaluate(self):
        metrics = ["%s_loss", "%s_acc"]
        evaluations = [self.loss / self.total, self.correct / self.total]
        self.log = ["%.4f" % eval for eval in evaluations]
        self.log_dict = {metric:eval for metric, eval in zip(metrics, evaluations)}

    def reset(self):
        self.total = 0.0
        self.correct = 0.0
        self.loss = 0.0
        self.log = []
        self.log_dit = {}


def get_optim_scheduler(params, cfg, batch_n):
    """ configure optim and scheduler """
    optim = torch.optim.SGD([{'params': params[0], 'weight_decay': cfg.weight_decay},
                             {'params': params[1], 'weight_decay': 0}],
                            lr=cfg.learning_rate, momentum=cfg.momentum, nesterov=True)
    if cfg.learning_rate_scheduler == "cyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=0, max_lr=cfg.learning_rate,
                                                      step_size_up=cfg.num_epochs * batch_n / 2,
                                                      step_size_down=cfg.num_epochs * batch_n / 2)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [step *batch_n for step in cfg.learning_rate_steps],
                                                         cfg.learning_rate_decay)

    return optim, scheduler

