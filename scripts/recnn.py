import glob
import argparse
import csv
import os
import random
import sys
import h5py
import numpy as np
from datetime import datetime
from models import *
from tqdm import tqdm, trange

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import trange
import torchvision
import torchvision.transforms as transforms

import flops_benchmark
from clr import CyclicLR
from data import get_loaders
from logger import CsvLogger
from run import train, test, save_checkpoint, find_bounds_clr
import scipy.io

parser = argparse.ArgumentParser(description='RECNN training with PyTorch')
parser.add_argument('--gpus', default=None, help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='Number of data loading workers (default: 4)')
parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')
parser.add_argument('--classes', type=int, default=100, help= "Number of catogories")

# Optimization options
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001, help='The learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=4e-5, help='Weight decay (L2 penalty).')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
parser.add_argument('--schedule', type=int, nargs='+', default=[30],
                    help='Decrease learning rate at these epochs.')

# CLR
parser.add_argument('--clr', dest='clr', action='store_true', help='Use CLR')
parser.add_argument('--min-lr', type=float, default=1e-5, help='Minimal LR for CLR.')
parser.add_argument('--max-lr', type=float, default=1, help='Maximal LR for CLR.')
parser.add_argument('--epochs-per-step', type=int, default=20,
                    help='Number of epochs per step in CLR, recommended to be between 2 and 10.')
parser.add_argument('--mode', default='triangular2', help='CLR mode. One of {triangular, triangular2, exp_range}')
parser.add_argument('--find-clr', dest='find_clr', action='store_true',
                    help='Run search for optimal LR in range (min_lr, max_lr)')

# Checkpoints
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Just evaluate model')
parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='Number of batches between log messages')
parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: 1)')
parser.add_argument('--h5dir', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

# Architecture
parser.add_argument('--scaling', type=float, default=1, metavar='SC', help='Scaling of RECNN (default x1).')
parser.add_argument('--input-size', type=int, default=224, metavar='I',
                    help='Input size of RECNN, multiple of 32 (default 224).')
parser.add_argument('--model', type=str, default='recnn', help='Model name.')


def main():
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus:
        torch.cuda.manual_seed_all(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = time_stamp
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.gpus is not None:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        device = 'cuda:' + str(args.gpus[0])
        cudnn.benchmark = True
    else:
        device = 'cpu'

    if args.type == 'float64':
        dtype = torch.float64
    elif args.type == 'float32':
        dtype = torch.float32
    elif args.type == 'float16':
        dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8

    if (args.model == "recnn"):
        print ("Training 3D RECNN")
        model = RECNN()
    else:
        print ("Error: no model matched!")
    num_parameters = sum([l.nelement() for l in model.parameters()])
    print(model)
    print('number of parameters: {}'.format(num_parameters))
    
    train_hdf5_list = [x for x in glob.glob(os.path.join(args.h5dir, 'train', '*.h5'))]
    tcnt = 0
    for f in train_hdf5_list:
        h5_file = h5py.File(f, 'r')
        tcnt = tcnt + 1
        if tcnt == 1:
            trainx = torch.from_numpy(np.array(h5_file['data']))
            trainy = torch.from_numpy(np.array(h5_file['label']))
        else:
            traincx = torch.from_numpy(np.array(h5_file['data']))
            traincy = torch.from_numpy(np.array(h5_file['label']))
            trainx = torch.cat((trainx, traincx), 0)
            trainy = torch.cat((trainy, traincy), 0)

    trx_shape = trainx.shape
    trainx = trainx.view(trx_shape[0],1,trx_shape[1],trx_shape[2], trx_shape[3])
    trainxy = torch.utils.data.TensorDataset(trainx, trainy)
    train_loader = torch.utils.data.DataLoader(trainxy, batch_size=args.batch_size, shuffle=True)

    test_hdf5_list = [x for x in glob.glob(os.path.join(args.h5dir, 'test', '*.h5'))]
    tcnt = 0
    for f in test_hdf5_list:
        h5_file = h5py.File(f, 'r')
        tcnt = tcnt + 1
        if tcnt == 1:
            testx = torch.from_numpy(np.array(h5_file['data']))
            testy = torch.from_numpy(np.array(h5_file['label']))
        else:
            testcx = torch.from_numpy(np.array(h5_file['data']))
            testcy = torch.from_numpy(np.array(h5_file['label']))
            testx = torch.cat((testx, testcx), 0)
            testy = torch.cat((testy, testcy), 0)

    tex_shape = testx.shape
    testx = testx.view(tex_shape[0],1,tex_shape[1],tex_shape[2], tex_shape[3])
    testxy = torch.utils.data.TensorDataset(testx, testy)
    val_loader = torch.utils.data.DataLoader(testxy, batch_size=args.batch_size, shuffle=False)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss()

    if args.gpus is not None:
        model = torch.nn.DataParallel(model, args.gpus)
    model.to(device=device, dtype=dtype)
    criterion.to(device=device, dtype=dtype)

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay,
                                nesterov=True)
    if args.find_clr:
        find_bounds_clr(model, train_loader, optimizer, criterion, device, dtype, min_lr=args.min_lr,
                        max_lr=args.max_lr, step_size=args.epochs_per_step * len(train_loader), mode=args.mode,
                        save_path=save_path)
        return

    if args.clr:
        scheduler = CyclicLR(optimizer, base_lr=args.min_lr, max_lr=args.max_lr,
                             step_size=args.epochs_per_step * len(train_loader), mode=args.mode)
    else:
        scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    best_test = 0

    # optionally resume from a checkpoint
    data = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch'] - 1
            best_test = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        elif os.path.isdir(args.resume):
            checkpoint_path = os.path.join(args.resume, 'checkpoint.pth.tar')
            csv_path = os.path.join(args.resume, 'results.csv')
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=device)
            args.start_epoch = checkpoint['epoch'] - 1
            best_test = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
            data = []
            with open(csv_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        loss, top1, top5 = test(model, val_loader, criterion, device, dtype)  # TODO
        return

    csv_logger = CsvLogger(filepath=save_path, data=data)
    csv_logger.save_params(sys.argv, args)

    claimed_acc1 = None
    claimed_acc5 = None
    best_test = 10000000
    train_network(args.start_epoch, args.epochs, scheduler, model, train_loader, val_loader, optimizer, criterion,
                  device, dtype, args.batch_size, args.log_interval, csv_logger, save_path, claimed_acc1, claimed_acc5,
                  best_test)


def train_network(start_epoch, epochs, scheduler, model, train_loader, val_loader, optimizer, criterion, device, dtype,
                  batch_size, log_interval, csv_logger, save_path, claimed_acc1, claimed_acc5, best_test):
    for epoch in trange(start_epoch, epochs + 1):
        if not isinstance(scheduler, CyclicLR):
            scheduler.step()
        train_loss = train(model, train_loader, epoch, optimizer, criterion, device,
                                                              dtype, batch_size, log_interval, scheduler)
        test_loss = test(model, val_loader, criterion, device, dtype)
        csv_logger.write({'epoch': epoch + 1, 'val_loss': test_loss, 'train_loss': train_loss})
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_test,
                         'optimizer': optimizer.state_dict()}, test_loss < best_test, filepath=save_path)
        if test_loss < best_test:
            best_test = test_loss

    csv_logger.write_text('Best loss is {}'.format(best_test))


if __name__ == '__main__':
    main()
