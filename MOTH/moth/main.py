# coding: utf-8

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import argparse
import numpy as np
import os
import sys
import time
import torch
import torch.nn.functional as F

from models import nin,vgg,resnet,wresnet,inception,densenet,mobilenetv2,efficientnet
from util import get_norm, preprocess, deprocess, pgd_attack
from util import get_num, get_size, get_dataloader, get_model
from inversion import Trigger, TriggerCombo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'


def moth():
    # assisting variables/parameters
    trigger_steps = 500
    warmup_steps  = 1
    cost   = 1e-3
    count  = np.zeros(2)
    WARMUP = True

    num_classes = get_num(args.dataset)
    img_rows, img_cols, img_channels = get_size(args.dataset)

    # matrices for recording distance changes
    mat_univ  = np.zeros((num_classes, num_classes)) # warmup distance
    mat_size  = np.zeros((num_classes, num_classes)) # trigger size
    mat_diff  = np.zeros((num_classes, num_classes)) # distance improvement
    mat_count = np.zeros((num_classes, num_classes)) # number of selected pairs

    mask_dict = {}
    pattern_dict = {}

    # load model
    model = resnet.resnet18(num_classes=10).to(args.device)
    state_dict = torch.load('logs/models/model_last_patch.th')
    model.load_state_dict(state_dict)
    model.train()
    # set loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                nesterov=True)

    # load data
    train_loader = get_dataloader(args.dataset,  True, args.data_ratio)
    test_loader  = get_dataloader(args.dataset, False, 0.05)

    # a subset for loss calculation during warmup
    for idx, (x_batch, y_batch) in enumerate(train_loader):
        if idx == 0:
            x_extra, y_extra = x_batch, y_batch
        else:
            x_extra = torch.cat((x_extra, x_batch))
            y_extra = torch.cat((y_extra, y_batch))
        if idx > 3:
            break

    num_samples = 10
    for i in range(num_classes):
        size = np.count_nonzero(y_extra == i)
        if size < num_samples:
            num_samples = size
    assert (num_samples > 0)

    indices = []
    for i in range(num_classes):
        idx = np.where(y_extra == i)[0]
        indices.extend(list(idx[:num_samples]))
    x_extra = x_extra[indices]
    y_extra = y_extra[indices]
    assert (x_extra.size(0) == num_samples * num_classes)

    # set up trigger generation
    trigger       = Trigger(
                        model,
                        args.dataset,
                        steps=trigger_steps,
                        asr_bound=0.99
                    )
    trigger_combo = TriggerCombo(
                        model,
                        args.dataset,
                        steps=trigger_steps
                    )

    bound_size = img_rows * img_cols * img_channels / 4

    if args.type == 'adv':
        # attack parameters
        if args.dataset == 'cifar10':
            epsilon, k, a = 8/255, 7, 2/255
        elif args.dataset in ['svhn', 'gtsrb']:
            epsilon, k, a = 0.03, 8, 0.005
        elif args.dataset == 'lisa':
            epsilon, k, a = 0.1, 8, 0.02

    # hardening iterations
    max_warmup_steps = warmup_steps * num_classes
    steps_per_epoch = len(train_loader)
    max_steps = max_warmup_steps + args.epochs * steps_per_epoch

    step = 0
    source, target = 0, -1

    # start hardening
    print('='*80)
    print('start hardening...')
    time_start = time.time()
    for epoch in range(args.epochs):
        for (x_batch, y_batch) in train_loader:
            x_batch = x_batch.to(args.device)

            if args.type == 'nat':
                x_adv = torch.clone(x_batch)
            elif args.type == 'adv':
                x_adv = pgd_attack(
                            model,
                            deprocess(x_batch, args.dataset),
                            y_batch.to(args.device),
                            args.mean,
                            args.std,
                            eps=epsilon,
                            alpha=a,
                            iters=k
                        )
                x_adv = preprocess(x_adv, args.dataset)

            # update variables after warmup stage
            if step >= max_warmup_steps:
                if WARMUP:
                    mat_diff /= np.max(mat_diff)
                WARMUP = False
                warmup_steps = 3

            # periodically update corresponding variables in each stage
            if (WARMUP and step % warmup_steps == 0) or\
               (not WARMUP and (step - max_warmup_steps) % warmup_steps == 0):
                if WARMUP:
                    target += 1
                    trigger_steps = 500
                else:
                    if np.random.rand() < 0.3:
                        # randomly select a pair
                        source, target = np.random.choice(
                                            np.arange(num_classes),
                                            2,
                                            replace=False
                                         )
                    else:
                        # select a pair according to distance improvement
                        univ_sum = mat_univ + mat_univ.transpose()
                        diff_sum = mat_diff + mat_diff.transpose()
                        alpha = np.minimum(
                                    0.1 * ((step - max_warmup_steps) / 100),
                                    1
                                )
                        diff_sum = (1 - alpha) * univ_sum + alpha * diff_sum
                        source, target = np.unravel_index(np.argmax(diff_sum),
                                                          diff_sum.shape)

                        print('-'*50)
                        print('fastest pair: {:d}-{:d}, improve: {:.2f}'\
                              .format(source, target, diff_sum[source, target]))

                    trigger_steps = 200

                if source < target:
                    key = f'{source}-{target}'
                else:
                    key = f'{target}-{source}'

                print('-'*50)
                print('selected pair:', key)

                # count the selected pair
                if not WARMUP:
                    mat_count[source, target] += 1
                    mat_count[target, source] += 1

                # use existing previous mask and pattern
                if key in mask_dict:
                    init_mask    = mask_dict[key]
                    init_pattern = pattern_dict[key]
                else:
                    init_mask    = None
                    init_pattern = None

                # reset values
                cost = 1e-3
                count[...] = 0
                mask_size_list = []

            if WARMUP:
                # get a few samples from each label
                indices = np.where(y_extra != target)[0]

                # trigger inversion set
                x_set = x_extra[indices]
                y_set = torch.full((x_set.shape[0],), target)

                # generate universal trigger
                mask, pattern, speed\
                        = trigger.generate(
                                (num_classes, target),
                                x_set,
                                y_set,
                                attack_size=len(indices),
                                steps=trigger_steps,
                                init_cost=cost,
                                init_m=init_mask,
                                init_p=init_pattern
                          )

                trigger_size = [mask.abs().sum().detach().cpu().numpy()] * 2

                if trigger_size[0] < bound_size:
                    # choose non-target samples to stamp the generated trigger
                    indices = np.where(y_batch != target)[0]
                    length = int(len(indices) * args.warm_ratio)
                    choice = np.random.choice(indices, length, replace=False)

                    # stamp trigger
                    x_batch_adv = (1 - mask)\
                                    * deprocess(x_batch[choice], args.dataset)\
                                        + mask * pattern
                    x_batch_adv = torch.clip(x_batch_adv, 0.0, 1.0)

                    x_adv[choice] = preprocess(x_batch_adv, args.dataset)

                mask    = mask.detach().cpu().numpy()
                pattern = pattern.detach().cpu().numpy()

                # record approximated distance improvement during warmup
                for i in range(num_classes):
                    # mean loss change of samples of each source label
                    if i < target:
                        diff = np.mean(speed[i*num_samples : (i+1)*num_samples])
                    elif i > target:
                        diff = np.mean(speed[(i-1)*num_samples : i*num_samples])

                    if i != target:
                        mat_univ[i, target] = diff

                        # save generated triggers of a pair
                        src, tgt = i, target
                        key = f'{src}-{tgt}' if src < tgt else f'{tgt}-{src}'
                        if key not in mask_dict:
                            mask_dict[key] = mask[:1, ...]
                            pattern_dict[key] = pattern
                        else:
                            if src < tgt:
                                mask_dict[key]    = np.stack(
                                                        [mask[:1, ...],
                                                            mask_dict[key]],
                                                        axis=0
                                                    )
                                pattern_dict[key] = np.stack(
                                                        [pattern,
                                                            pattern_dict[key]],
                                                        axis=0
                                                    )
                            else:
                                mask_dict[key]    = np.stack(
                                                        [mask_dict[key],
                                                            mask[:1, ...]],
                                                        axis=0
                                                    )
                                pattern_dict[key] = np.stack(
                                                        [pattern_dict[key],
                                                            pattern],
                                                        axis=0
                                                    )

                        # initialize distance matrix entries
                        mat_size[i, target] = trigger_size[0]
                        mat_diff[i, target] = mat_size[i, target]
            else:
                # get samples from source and target labels
                idx_source = np.where(y_batch == source)[0]
                idx_target = np.where(y_batch == target)[0]

                # use a portion of source/target samples
                length = int(min(len(idx_source), len(idx_target))\
                                * args.portion)
                if length > 0:
                    # dynamically adjust parameters
                    if (step - max_warmup_steps) % warmup_steps > 0:
                        if count[0] > 0 or count[1] > 0:
                            trigger_steps = 200
                            cost = 1e-3
                            count[...] = 0
                        else:
                            trigger_steps = 50
                            cost = 1e-2

                    # construct generation set for both directions
                    # source samples with target labels
                    # target samples with source labels
                    x_set = torch.cat((x_batch[idx_source],
                                       x_batch[idx_target]))
                    y_target = torch.full((len(idx_source),), target)
                    y_source = torch.full((len(idx_target),), source)
                    y_set = torch.cat((y_target, y_source))

                    # indicator vector for source/target
                    m_set = torch.zeros(x_set.shape[0])
                    m_set[:len(idx_source)] = 1

                    # generate a pair of triggers
                    mask, pattern\
                            = trigger_combo.generate(
                                    (source, target),
                                    x_set,
                                    y_set,
                                    m_set,
                                    attack_size=x_set.shape[0],
                                    steps=trigger_steps,
                                    init_cost=cost,
                                    init_m=init_mask,
                                    init_p=init_pattern
                              )

                    trigger_size = mask.abs().sum(axis=(1, 2, 3)).detach()\
                                                                 .cpu().numpy()

                    # operate on two directions
                    for cb in range(2):
                        if trigger_size[cb] < bound_size:
                            # choose samples to stamp the generated trigger
                            indices = idx_source if cb == 0 else idx_target
                            choice = np.random.choice(indices, length,
                                                      replace=False)

                            # stamp trigger
                            x_batch_adv\
                                = (1 - mask[cb])\
                                    * deprocess(x_batch[choice], args.dataset)\
                                        + mask[cb] * pattern[cb]
                            x_batch_adv = torch.clip(x_batch_adv, 0.0, 1.0)

                            x_adv[choice] = preprocess(x_batch_adv, args.dataset)

                    # save generated triggers of a pair
                    mask    = mask.detach().cpu().numpy()
                    pattern = pattern.detach().cpu().numpy()
                    for cb in range(2):
                        if init_mask is None:
                            init_mask    = mask[:, :1, ...]
                            init_pattern = pattern

                            if key not in mask_dict:
                                mask_dict[key]    = init_mask
                                pattern_dict[key] = init_pattern
                        else:
                            if np.sum(mask[cb]) > 0:
                                init_mask[cb]    = mask[cb, :1, ...]
                                init_pattern[cb] = pattern[cb]
                                # save large trigger
                                if np.sum(init_mask[cb])\
                                        > np.sum(mask_dict[key][cb]):
                                    mask_dict[key][cb]    = init_mask[cb]
                                    pattern_dict[key][cb] = init_pattern[cb]
                            else:
                                # record failed generation
                                count[cb] += 1

                    mask_size_list.append(
                            list(np.sum(3 * np.abs(init_mask), axis=(1, 2, 3)))
                    )

                # periodically update distance related matrices
                if (step - max_warmup_steps) % warmup_steps == warmup_steps - 1:
                    if len(mask_size_list) <= 0:
                        continue

                    # average trigger size of the current hardening period
                    mask_size_avg = np.mean(mask_size_list, axis=0)
                    if mat_size[source, target] == 0\
                            or mat_size[target, source] == 0:
                        mat_size[source, target] = mask_size_avg[0]
                        mat_size[target, source] = mask_size_avg[1]
                        mat_diff = mat_size
                        mat_diff[mat_diff == -1] = 0
                    else:
                        # compute distance improvement
                        last_warm = mat_size[source, target]
                        mat_diff[source, target]\
                                += (mask_size_avg[0] - last_warm) / last_warm
                        mat_diff[source, target] /= 2

                        last_warm = mat_size[target, source]
                        mat_diff[target, source]\
                                += (mask_size_avg[1] - last_warm) / last_warm
                        mat_diff[target, source] /= 2

                        # update recorded trigger size
                        mat_size[source, target] = mask_size_avg[0]
                        mat_size[target, source] = mask_size_avg[1]

            x_batch = x_adv.detach()

            # train model
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch.to(args.device))
            loss.backward()
            optimizer.step()

            # evaluate and save model
            if (step+1) % 10 == 0:
                time_end = time.time()

                total = 0
                correct = 0
                with torch.no_grad():
                    for (x_test, y_test) in test_loader:
                        x_test = x_test.to(args.device)
                        y_test = y_test.to(args.device)
                        total += y_test.size(0)

                        y_out = model(x_test)
                        _, y_pred = torch.max(y_out.data, 1)
                        correct += (y_pred == y_test).sum().item()
                acc = correct / total

                time_cost = time_end - time_start
                print('*'*120)
                sys.stdout.write('step: {:4}/{:4} - {:.2f}s, '\
                                    .format(step+1, max_steps, time_cost)\
                                 + 'loss: {:.4f}, acc: {:.4f}\t'
                                    .format(loss, acc)\
                                 + 'trigger size: ({:.2f}, {:.2f})\n'
                                    .format(trigger_size[0], trigger_size[1]))
                sys.stdout.flush()
                print('*'*120)

                save_name = f'{args.dataset}_{args.model}_{args.suffix}_moth'
                np.save(f'data/pair_count/{save_name}', mat_count)
                torch.save(model.state_dict(), f'ckpt/{save_name}.pt')

                time_start = time.time()

            if step + 1 >= max_steps:
                break

            step += 1

        if step + 1 >= max_steps:
            break

    np.save(f'data/pair_count/{save_name}', mat_count)
    torch.save(model.state_dict(), f'ckpt/{save_name}.pt')


def test():
    model = get_model(args)
    path = f'ckpt/{args.dataset}_{args.model}_{args.suffix}.pt'
    model.load_state_dict(torch.load(path))
    model.to(args.device)
    model.eval()

    test_loader = get_dataloader(args.dataset, False)

    total   = 0
    correct = 0
    print('-'*50)
    for (x_test, y_test) in test_loader:
        with torch.no_grad():
            x_test, y_test = x_test.to(args.device), y_test.to(args.device)
            total += x_test.shape[0]

            pred = model(x_test)
            correct += torch.sum(torch.argmax(pred, 1) == y_test)

            acc = correct / total

            sys.stdout.write('\racc: {:.4f}'.format(acc))
            sys.stdout.flush()
    print()


def measure():
    # load model
    model = get_model(args)
    path = f'ckpt/{args.dataset}_{args.model}_{args.suffix}.pt'
    model.load_state_dict(torch.load(path))
    model.to(args.device)
    model.eval()

    # load data
    size = 100
    test_loader = get_dataloader(args.dataset, False, batch_size=size)
    num_classes = get_num(args.dataset)

    # attack parameters
    if args.dataset == 'cifar10':
        epsilon, k, a = 8/255, 20, 2/255
    elif args.dataset in ['svhn', 'gtsrb']:
        epsilon, k, a = 0.03, 100, 0.001
    elif args.dataset == 'lisa':
        epsilon, k, a = 0.1, 100, 0.01

    # generate adversarial examples
    results = np.zeros((num_classes, num_classes), dtype=int)
    print('-'*50)
    for start in range(0, 10):
        sys.stdout.write(f'\rattacking batch {start}')
        sys.stdout.flush()

        # get a batch
        inputs, labels = next(iter(test_loader))
        inputs = deprocess(inputs.to(args.device), args.dataset)
        if len(inputs) != size:
            break

        # generate adv
        adv = pgd_attack(
                  model,
                  inputs,
                  labels.to(args.device),
                  args.mean,
                  args.std,
                  eps=epsilon,
                  alpha=a,
                  iters=k
              )

        pred_adv = model(preprocess(adv, args.dataset))

        # record prediction
        for i in range(size):
            id_ori = labels[i]
            id_adv = torch.argmax(pred_adv[i])
            results[id_ori, id_adv] += 1
    print()

    # show robustness
    robust = []
    total = 0
    print('-'*80)
    for i in range(num_classes):
        for j in range(num_classes):
            print(results[i, j], end='\t')
            if i == j:
                robust.append(results[i, j] / np.sum(results[i]))
                total += results[i, j]
        print()
    print('-'*80)

    print('\n'.join(f'{i}: {robust[i]}' for i in range(len(robust))))
    print('-'*50)
    print(f'robustness: {total / np.sum(results)}')


def validate():
    prefix = f'{args.dataset}_{args.model}_{args.suffix}'

    # load model
    model = get_model(args)
    path = f'ckpt/{args.dataset}_{args.model}_{args.suffix}.pt'
    model.load_state_dict(torch.load(path))
    model.to(args.device)
    model.eval()

    # load data
    test_loader = get_dataloader(args.dataset, False)
    num_classes = get_num(args.dataset)

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if batch_idx == 0:
            x_val, y_val = inputs, targets
        else:
            x_val = torch.cat((x_val, inputs))
            y_val = torch.cat((y_val, targets))

        if len(x_val) >= num_classes * 100:
            break
    print(x_val.shape, y_val.shape)

    # initialize trigger generation
    trigger = Trigger(model, args.dataset)

    print('-'*80)
    print('validating pair distance...')
    print('-'*80)
    if args.pair != '0-0':
        # generate triggers for one class pair
        source, target = list(map(int, args.pair.split('-')))
        mask, pattern, _ = trigger.generate((source, target), x_val, y_val)
        size = mask.abs().sum().detach().cpu().numpy()
        print(f'distance for {source}->{target}: {size:.2f}')
    else:
        fsave = open(f'data/distance/{prefix}_{args.seed}.txt', 'a')

        # generate triggers for all class pairs
        for source in range(num_classes):
            for target in range(num_classes):
                if source != target:
                    mask, pattern, _\
                            = trigger.generate((source, target), x_val, y_val)
                    size = mask.abs().sum().detach().cpu().numpy()

                    fsave.write(str(size) + ',')
                    fsave.flush()
            fsave.write('\n')


def show():
    prefix = f'data/distance/{args.dataset}_{args.model}'

    data = []
    for i in range(3):
        data_seed = []
        for line in open(f'{prefix}_{args.suffix}_{i}.txt', 'r'):
            line = line.strip().split(',')[:-1]
            line = list(map(float, line))
            data_seed.append(line)
        data.append(data_seed)

    data = np.array(data)
    data[data == 0] = np.inf
    data = np.min(data, axis=0)
    data[data == np.inf] = 0
    np.save(f'{prefix}_{args.suffix}', data)

    print('-'*100)
    for i in range(num_classes):
        for j in range(num_classes):
            if j < i:
                print(str(data[i, j]), end='\t')
            elif j == i:
                print('-', end='\t')
            else:
                print(str(data[i, j-1]), end='\t')
        print()
    print('-'*100)
    print(f'average distance: {np.mean(data):.2f}')

    if 'nat' in args.suffix:
        base = np.load(f'{prefix}_nat.npy')
    elif 'adv' in args.suffix:
        base = np.load(f'{prefix}_adv.npy')
    diff = np.mean((data - base) / base)
    print(f'increase percent: {diff*100:.2f}%')



################################################################
############                  main                  ############
################################################################
def main():
    if args.phase == 'moth':
        moth()
    elif args.phase == 'test':
        test()
    elif args.phase == 'measure':
        measure()
    elif args.phase == 'validate':
        validate()
    elif args.phase == 'show':
        show()
    else:
        print('Option [{}] is not supported!'.format(args.phase))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')

    parser.add_argument('--gpu',     default='0',        help='gpu id')
    parser.add_argument('--seed',    default=1,          help='seed index', type=int)

    parser.add_argument('--phase',   default='validate',     help='phase of framework')
    parser.add_argument('--dataset', default='cifar10',  help='dataset')
    parser.add_argument('--model',   default='resnet18', help='model')
    parser.add_argument('--type',    default='nat',      help='model type (natural or adversarial)')
    parser.add_argument('--suffix',  default='nat',      help='checkpoint path')
    parser.add_argument('--pair',    default='0-0',      help='label pair')

    parser.add_argument('--batch_size', default=128,  type=int,   help='batch size')
    parser.add_argument('--epochs',     default=2,    type=int,   help='hardening epochs')
    parser.add_argument('--lr',         default=1e-4, type=float, help='learning rate')
    parser.add_argument('--data_ratio', default=1.0,  type=float, help='ratio of training samples for hardening')
    parser.add_argument('--warm_ratio', default=0.5,  type=float, help='ratio of batch samples to stamp trigger during warmup')
    parser.add_argument('--portion',    default=0.1,  type=float, help='ratio of batch samples to stamp trigger during orthogonalization')

    args = parser.parse_args()

    # set gpu usage
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # set random seed
    SEED = [1024, 557540351, 157301989]
    SEED = SEED[args.seed]
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # set basics
    args.device = torch.device('cuda')
    args.mean, args.std = get_norm(args.dataset)

    # main function
    time_start = time.time()
    main()
    time_end = time.time()
    print('='*50)
    print('Running time:', (time_end - time_start) / 60, 'm')
    print('='*50)
