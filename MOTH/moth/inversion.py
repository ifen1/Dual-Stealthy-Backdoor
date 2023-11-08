# coding: utf-8

import numpy as np
import torch
import sys

from util import get_num, get_size, preprocess, deprocess


class Trigger:
    def __init__(self,
                 model,             # subject model
                 dataset,           # dataset
                 steps=1000,        # number of steps for trigger inversion
                 batch_size=32,     # batch size in trigger inversion
                 asr_bound=0.9      # threshold for attack success rate
        ):                          # maximum pixel value

        self.model = model
        self.dataset = dataset
        self.steps = steps
        self.batch_size = batch_size
        self.asr_bound = asr_bound

        self.device = torch.device('cuda')
        self.num_classes = get_num(dataset)
        self.img_rows, self.img_cols, self.img_channels = get_size(dataset)

        # hyper-parameters to dynamically adjust loss weight
        self.epsilon = 1e-7
        self.patience = 10
        self.cost_multiplier_up   = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5

        self.mask_size    = [self.img_rows, self.img_cols]
        self.pattern_size = [self.img_channels, self.img_rows, self.img_cols]

    def generate(self, pair, x_set, y_set, attack_size=100, steps=1000,
                 init_cost=1e-3, init_m=None, init_p=None):
        source, target = pair

        # update hyper-parameters
        self.steps = steps
        self.batch_size = np.minimum(self.batch_size, attack_size)

        # store best results
        mask_best    = torch.zeros(self.pattern_size).to(self.device)
        pattern_best = torch.zeros(self.pattern_size).to(self.device)
        reg_best     = float('inf')

        # hyper-parameters to dynamically adjust loss weight
        cost = init_cost
        cost_up_counter   = 0
        cost_down_counter = 0

        # initialize mask and pattern
        if init_m is None:
            init_mask = np.random.random(self.mask_size)
        else:
            init_mask = init_m

        if init_p is None:
            init_pattern = np.random.random(self.pattern_size)
        else:
            init_pattern = init_p

        init_mask    = np.clip(init_mask, 0.0, 1.0)
        init_mask    = np.arctanh((init_mask - 0.5) * (2 - self.epsilon))
        init_pattern = np.clip(init_pattern, 0.0, 1.0)
        init_pattern = np.arctanh((init_pattern - 0.5) * (2 - self.epsilon))

        # set mask and pattern variables with init values
        self.mask_tensor    = torch.Tensor(init_mask).to(self.device)
        self.pattern_tensor = torch.Tensor(init_pattern).to(self.device)
        self.mask_tensor.requires_grad    = True
        self.pattern_tensor.requires_grad = True

        # select inputs for label-specific or universal attack
        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set == target)[0]
            if indices.shape[0] != y_set.shape[0]:
                indices = np.where(y_set != target)[0]

            # record loss change
            loss_start = np.zeros(x_set.shape[0])
            loss_end   = np.zeros(x_set.shape[0])

        # choose a subset of samples for trigger inversion
        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].to(self.device)
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        # avoid having the number of inputs smaller than batch size
        self.batch_size = np.minimum(self.batch_size, x_set.shape[0])

        # set loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam([self.mask_tensor, self.pattern_tensor],
                                     lr=0.1, betas=(0.5, 0.9))

        # record samples' indices during suffling
        index_base = np.arange(x_set.shape[0])

        # start generation
        self.model.eval()
        for step in range(self.steps):
            # shuffle training samples
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]
            index_base = index_base[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(int(np.ceil(x_set.shape[0] / self.batch_size))):
                # get a batch of data
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]
                x_batch = deprocess(x_batch, self.dataset, clone=False)

                # define mask and pattern
                self.mask = (torch.tanh(self.mask_tensor)\
                                / (2 - self.epsilon) + 0.5)\
                                    .repeat(self.img_channels, 1, 1)
                self.pattern = torch.tanh(self.pattern_tensor)\
                                / (2 - self.epsilon) + 0.5

                # stamp trigger pattern
                x_adv = (1 - self.mask) * x_batch + self.mask * self.pattern

                optimizer.zero_grad()

                output = self.model(preprocess(x_adv, self.dataset, clone=False))

                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item()\
                            / x_batch.shape[0]

                # loss
                loss_ce  = criterion(output, y_batch)
                loss_reg = torch.sum(torch.abs(self.mask)) / self.img_channels
                loss = loss_ce.mean() + loss_reg * cost

                loss.backward()
                optimizer.step()

                # record loss and accuracy
                loss_ce_list.extend( loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(    loss.detach().cpu().numpy())
                acc_list.append(     acc)

            # record the initial loss value
            if source == self.num_classes\
                    and step == 0\
                    and len(loss_ce_list) == attack_size:
                loss_start[index_base] = loss_ce_list

            # calculate average loss and accuracy
            avg_loss_ce  = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss     = np.mean(loss_list)
            avg_acc      = np.mean(acc_list)

            # record the best mask and pattern
            if avg_acc >= self.asr_bound and avg_loss_reg < reg_best:
                mask_best    = self.mask
                pattern_best = self.pattern
                reg_best     = avg_loss_reg

                # add samll perturbations to mask and pattern
                # to avoid stucking in local minima
                epsilon = 0.01
                init_mask    = mask_best[0, ...]
                init_mask    = init_mask + torch.distributions.Uniform(\
                                    low=-epsilon, high=epsilon)\
                                        .sample(init_mask.shape).to(self.device)
                init_mask    = torch.clip(init_mask, 0.0, 1.0)
                init_mask    = torch.arctanh((init_mask - 0.5)\
                                                * (2 - self.epsilon))
                init_pattern = pattern_best + torch.distributions.Uniform(\
                                    low=-epsilon, high=epsilon)\
                                        .sample(init_pattern.shape)\
                                            .to(self.device)
                init_pattern = torch.clip(init_pattern, 0.0, 1.0)
                init_pattern = torch.arctanh((init_pattern - 0.5)\
                                                * (2 - self.epsilon))

                with torch.no_grad():
                    self.mask_tensor.copy_(init_mask)
                    self.pattern_tensor.copy_(init_pattern)

                # record the final loss value when the best trigger is saved
                if source == self.num_classes\
                        and loss_ce.shape[0] == attack_size:
                    loss_end[index_base] = loss_ce.detach().cpu().numpy()

            # helper variables for adjusting loss weight
            if avg_acc >= self.asr_bound:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            # adjust loss weight
            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if cost == 0:
                    cost = init_cost
                else:
                    cost *= self.cost_multiplier_up
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                cost /= self.cost_multiplier_down

            # periodically print inversion results
            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                    .format(step, avg_acc, avg_loss)
                                 + 'ce: {:.2f}, reg: {:.2f}, reg_best: {:.2f}  '\
                                    .format(avg_loss_ce, avg_loss_reg, reg_best))
                sys.stdout.flush()

        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rmask norm of pair {:d}-{:d}: {:.2f}\n'\
                            .format(source, target, mask_best.abs().sum()))
        sys.stdout.flush()

        # compute loss difference
        if source == self.num_classes and len(loss_ce_list) == attack_size:
            indices = np.where(loss_start == 0)[0]
            loss_start[indices] = 1
            loss_monitor = (loss_start - loss_end) / loss_start
            loss_monitor[indices] = 0
        else:
            loss_monitor = np.zeros(x_set.shape[0])

        return mask_best, pattern_best, loss_monitor


class TriggerCombo:
    def __init__(self,
                 model,             # subject model
                 dataset,           # dataset
                 steps=1000,        # number of steps for trigger inversion
                 batch_size=32,     # batch size in trigger inversion
                 asr_bound=0.9,     # threshold for attack success rate
        ):

        self.model = model
        self.dataset = dataset
        self.steps = steps
        self.batch_size = batch_size
        self.asr_bound = asr_bound

        self.device = torch.device('cuda')
        self.img_rows, self.img_cols, self.img_channels = get_size(dataset)

        # hyper-parameters to dynamically adjust loss weight
        self.epsilon = 1e-7
        self.patience = 10
        self.cost_multiplier_up   = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5

        self.mask_size    = [2, 1, self.img_rows, self.img_cols]
        self.pattern_size = [2, self.img_channels, self.img_rows, self.img_cols]

    def generate(self, pair, x_set, y_set, m_set, attack_size=100, steps=1000,
                 init_cost=1e-3, init_m=None, init_p=None):
        source, target = pair

        # update hyper-parameters
        self.steps = steps
        self.batch_size = np.minimum(self.batch_size, attack_size)

        # store best results
        mask_best    = torch.zeros(self.pattern_size).to(self.device)
        pattern_best = torch.zeros(self.pattern_size).to(self.device)
        reg_best     = [float('inf')] * 2

        # hyper-parameters to dynamically adjust loss weight
        cost = [init_cost] * 2
        cost_up_counter   = [0] * 2
        cost_down_counter = [0] * 2

        # initialize mask and pattern
        if init_m is None:
            init_mask = np.random.random(self.mask_size)
        else:
            init_mask = init_m

        if init_p is None:
            init_pattern = np.random.random(self.pattern_size)
        else:
            init_pattern = init_p

        init_mask    = np.clip(init_mask, 0.0, 1.0)
        init_mask    = np.arctanh((init_mask - 0.5) * (2 - self.epsilon))
        init_pattern = np.clip(init_pattern, 0.0, 1.0)
        init_pattern = np.arctanh((init_pattern - 0.5) * (2 - self.epsilon))

        # set mask and pattern variables with init values
        self.mask_tensor    = torch.Tensor(init_mask).to(self.device)
        self.pattern_tensor = torch.Tensor(init_pattern).to(self.device)
        self.mask_tensor.requires_grad    = True
        self.pattern_tensor.requires_grad = True

        # set loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam([self.mask_tensor, self.pattern_tensor],
                                     lr=0.1, betas=(0.5, 0.9))

        self.model.eval()
        x_set = x_set.to(self.device)
        y_set = y_set.to(self.device)
        m_set = m_set.to(self.device)
        for step in range(self.steps):
            # shuffle training samples
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]
            m_set = m_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                # get a batch of data
                x_batch = x_set[idx * self.batch_size : (idx+1) * self.batch_size]
                y_batch = y_set[idx * self.batch_size : (idx+1) * self.batch_size]
                m_batch = m_set[idx * self.batch_size : (idx+1) * self.batch_size]
                x_batch = deprocess(x_batch, self.dataset, clone=False)

                # define mask and pattern
                self.mask = (torch.tanh(self.mask_tensor)\
                                / (2 - self.epsilon) + 0.5)\
                                    .repeat(1, self.img_channels, 1, 1)
                self.pattern = torch.tanh(self.pattern_tensor)\
                                / (2 - self.epsilon) + 0.5

                # stamp trigger patterns for different pair directions
                x_adv = m_batch[:, None, None, None]\
                            * ((1 - self.mask[0]) * x_batch\
                                    + self.mask[0] * self.pattern[0])\
                        + (1 - m_batch[:, None, None, None])\
                            * ((1 - self.mask[1]) * x_batch\
                                    + self.mask[1] * self.pattern[1])

                optimizer.zero_grad()

                output = self.model(preprocess(x_adv, self.dataset, clone=False))

                # attack accuracy
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).squeeze()
                acc = [((m_batch * acc).sum()\
                            / m_batch.sum()).detach().cpu().numpy(),\
                       (((1 - m_batch) * acc).sum()\
                            / (1 - m_batch).sum()).detach().cpu().numpy()
                      ]

                # cross entropy loss
                loss_ce = criterion(output, y_batch)
                loss_ce_0 = (m_batch * loss_ce).sum().to(self.device)
                loss_ce_1 = ((1 - m_batch) * loss_ce).sum().to(self.device)

                # trigger size loss
                loss_reg = torch.sum(torch.abs(self.mask), dim=(1, 2, 3))\
                                / self.img_channels

                # total loss
                loss_0 = loss_ce_0 + loss_reg[0] * cost[0]
                loss_1 = loss_ce_1 + loss_reg[1] * cost[1]
                loss = loss_0 + loss_1

                loss.backward()
                optimizer.step()

                # record loss and accuracy
                loss_ce_list.append([loss_ce_0.detach().cpu().numpy(),\
                                     loss_ce_1.detach().cpu().numpy()])
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(   [loss_0.detach().cpu().numpy(),\
                                     loss_1.detach().cpu().numpy()])
                acc_list.append(acc)

            # calculate average loss and accuracy
            avg_loss_ce  = np.mean(loss_ce_list,  axis=0)
            avg_loss_reg = np.mean(loss_reg_list, axis=0)
            avg_loss     = np.mean(loss_list,     axis=0)
            avg_acc      = np.mean(acc_list,      axis=0)

            # update results for two directions of a pair
            for cb in range(2):
                # record the best mask and pattern
                if avg_acc[cb] >= self.asr_bound\
                        and avg_loss_reg[cb] < reg_best[cb]:
                    mask_best_local    = self.mask
                    mask_best[cb]      = mask_best_local[cb]
                    pattern_best_local = self.pattern
                    pattern_best[cb]   = pattern_best_local[cb]
                    reg_best[cb]       = avg_loss_reg[cb]

                    # add samll perturbations to mask and pattern
                    # to avoid stucking in local minima
                    epsilon = 0.01
                    init_mask    = mask_best_local[cb, :1, ...]
                    init_mask    = init_mask + torch.distributions.Uniform(\
                                        low=-epsilon, high=epsilon)\
                                            .sample(init_mask.shape)\
                                                .to(self.device)
                    init_pattern = pattern_best_local[cb]
                    init_pattern = init_pattern + torch.distributions.Uniform(\
                                        low=-epsilon, high=epsilon)\
                                            .sample(init_pattern.shape)\
                                                .to(self.device)

                    # stack mask and pattern in the corresponding direction
                    otr_idx = (cb + 1) % 2
                    if cb == 0:
                        init_mask    = torch.stack([
                                            init_mask,
                                            mask_best_local[otr_idx][:1, ...]
                                       ])
                        init_pattern = torch.stack([
                                            init_pattern,
                                            pattern_best_local[otr_idx]
                                       ])
                    else:
                        init_mask    = torch.stack([
                                            mask_best_local[otr_idx][:1, ...],
                                            init_mask
                                       ])
                        init_pattern = torch.stack([
                                            pattern_best_local[otr_idx],
                                            init_pattern
                                       ])

                    init_mask    = torch.clip(init_mask, 0.0, 1.0)
                    init_mask    = torch.arctanh((init_mask - 0.5)\
                                                    * (2 - self.epsilon))
                    init_pattern = torch.clip(init_pattern, 0.0, 1.0)
                    init_pattern = torch.arctanh((init_pattern - 0.5)\
                                                    * (2 - self.epsilon))

                    with torch.no_grad():
                        self.mask_tensor.copy_(init_mask)
                        self.pattern_tensor.copy_(init_pattern)

                # helper variables for adjusting loss weight
                if avg_acc[cb] >= self.asr_bound:
                    cost_up_counter[cb] += 1
                    cost_down_counter[cb] = 0
                else:
                    cost_up_counter[cb] = 0
                    cost_down_counter[cb] += 1

                # adjust loss weight
                if cost_up_counter[cb] >= self.patience:
                    cost_up_counter[cb] = 0
                    if cost[cb] == 0:
                        cost[cb] = init_cost
                    else:
                        cost[cb] *= self.cost_multiplier_up
                elif cost_down_counter[cb] >= self.patience:
                    cost_down_counter[cb] = 0
                    cost[cb] /= self.cost_multiplier_down

            # periodically print inversion results
            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: ({:.2f}, {:.2f}), '\
                                    .format(step, avg_acc[0], avg_acc[1])
                                 + 'loss: ({:.2f}, {:.2f}), '\
                                    .format(avg_loss[0], avg_loss[1])
                                 + 'ce: ({:.2f}, {:.2f}), '\
                                    .format(avg_loss_ce[0], avg_loss_ce[1])
                                 + 'reg: ({:.2f}, {:.2f}), '\
                                    .format(avg_loss_reg[0], avg_loss_reg[1])
                                 + 'reg_best: ({:.2f}, {:.2f})  '\
                                    .format(reg_best[0], reg_best[1]))
                sys.stdout.flush()

        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rmask norm of pair {:d}-{:d}: {:.2f}\n'\
                            .format(source, target, mask_best[0].abs().sum()))
        sys.stdout.write('\rmask norm of pair {:d}-{:d}: {:.2f}\n'\
                            .format(target, source, mask_best[1].abs().sum()))
        sys.stdout.flush()

        return mask_best, pattern_best
