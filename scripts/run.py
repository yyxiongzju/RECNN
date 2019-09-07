import os
import shutil

import matplotlib
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm, trange

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from clr import CyclicLR


def train(model, loader, epoch, optimizer, criterion, device, dtype, batch_size, log_interval, scheduler):
    model.train()
    correct1, correct5 = 0, 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        if isinstance(scheduler, CyclicLR):
            scheduler.batch_step()
        data, target = data.to(device=device, dtype=dtype), target.to(device=device, dtype=dtype)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()


        if batch_idx % log_interval == 0:
            tqdm.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. '.format(epoch, batch_idx, len(loader),
                                                           100. * batch_idx / len(loader), loss.item()))
    return loss.item()


def test(model, loader, criterion, device, dtype):
    model.eval()
    test_loss = 0
    correct1, correct5 = 0, 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device, dtype=dtype)
        with torch.no_grad():
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            #corr = correct(output, target, topk=(1, 5))
        #correct1 += corr[0]
        #correct5 += corr[1]

    test_loss /= len(loader)

    tqdm.write(
        '\nTest set: Average loss: {:.4f} '.format(test_loss))
    return test_loss

def extract_features(model, ex_model, loader, criterion, device, dtype):
    model.eval()
    ex_model.eval()
    test_loss = 0
    correct1, correct5 = 0, 0

    features = None
    predictions = None
    groundth = None
    cnt = 0
    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device, dtype=dtype)
        with torch.no_grad():
            output = model(data)
            ft_output = ex_model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            cnt = cnt + 1
            if cnt == 1:
                predictions = output
                features = ft_output
                groundth = target
            else:
                predictions = torch.cat((predictions, output), 0)
                features = torch.cat((features, ft_output), 0)
                groundth = torch.cat((groundth, target), 0)

            #corr = correct(output, target, topk=(1, 5))
        #correct1 += corr[0]
        #correct5 += corr[1]
    test_loss /= len(loader)

    tqdm.write(
        '\nTest set: Average loss: {:.4f} '.format(test_loss))
    return (features, predictions, groundth)

def extract_masks(model, ex_model, loader, criterion, device, dtype):
    model.eval()
    ex_model.eval()
    test_loss = 0
    correct1, correct5 = 0, 0

    features = None
    predictions = None
    groundth = None
    cnt = 0
    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device, dtype=dtype)
        with torch.no_grad():
            output = model(data)
            ft_output = ex_model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            cnt = cnt + 1
            if cnt == 1:
                predictions = output
                features = ft_output
                groundth = target
            else:
                predictions = torch.cat((predictions, output), 0)
                features = torch.cat((features, ft_output), 0)
                groundth = torch.cat((groundth, target), 0)

            #corr = correct(output, target, topk=(1, 5))
        #correct1 += corr[0]
        #correct5 += corr[1]
    test_loss /= len(loader)

    tqdm.write(
        '\nTest set: Average loss: {:.4f} '.format(test_loss))
    return (features, predictions, groundth)

def correct(output, target, topk=(1,)):
    """Computes the correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k)
    return res


def save_checkpoint(state, is_best, filepath='./', filename='checkpoint.pth.tar'):
    save_path = os.path.join(filepath, filename)
    best_path = os.path.join(filepath, 'model_best.pth.tar')
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, best_path)


def find_bounds_clr(model, loader, optimizer, criterion, device, dtype, min_lr=8e-6, max_lr=8e-5, step_size=2000,
                    mode='triangular', save_path='.'):
    model.train()
    correct1, correct5 = 0, 0
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size=step_size, mode=mode)
    epoch_count = step_size // len(loader)  # Assuming step_size is multiple of batch per epoch
    accuracy = []
    for _ in trange(epoch_count):
        for batch_idx, (data, target) in enumerate(tqdm(loader)):
            if scheduler is not None:
                scheduler.batch_step()
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            corr = correct(output, target)
            accuracy.append(corr[0] / data.shape[0])

    lrs = np.linspace(min_lr, max_lr, step_size)
    plt.plot(lrs, accuracy)
    plt.show()
    plt.savefig(os.path.join(save_path, 'find_bounds_clr.png'))
    np.save(os.path.join(save_path, 'acc.npy'), accuracy)
    return
