import matlab.engine

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import *
from dataset.SpinalDataset import *

from utils import Bar, Logger, AverageMeter, normalizedME, mkdir_p, savefig
from utils.cobb import *
import pandas as pd

parser = argparse.ArgumentParser(description='Spinal landmark Training')
# Datasets
parser.add_argument('-d', '--dataset', default='Spine', type=str)
parser.add_argument('-p', '--datapath', default='/home/felix/data/AASCE/boostnet_labeldata/', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=64, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0.5, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[100,200],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.2, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint/00/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='/home/felix/work/spine/spinal_landmark_estimation/checkpoint/00/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--depth', type=int, default=104, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc= 999  # best test accuracy

def main():

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_test = transforms.Compose([
        #SmartRandomCrop(),
        Rescale((256, 128)),
        ToTensor(),
        #Normalize([ 0.485, 0.485, 0.485,], [ 0.229, 0.229, 0.229,]),
    ])

    testset = SpinalDataset(
        csv_file = args.datapath + '/labels/test/filenames.csv', transform=transform_test,
        img_dir = args.datapath + '/data/test/', landmark_dir = args.datapath + '/labels/test/')
    testloader = data.DataLoader(testset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    model = resnet(136)
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.MSELoss().cuda()

    ignored_params = list(map(id, model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())
    params = [
        {'params': base_params, 'lr': args.lr},
        {'params': model.fc.parameters(), 'lr': args.lr * 10}
    ]
    model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(params=params, lr=args.lr, weight_decay=args.weight_decay)

    # Resume
    title = 'facelandmark_squeezenet_64'

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    args.checkpoint = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if os.path.exists(os.path.join(args.checkpoint, title+'_log.txt')):
        logger = Logger(os.path.join(args.checkpoint, title+'_log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, title+'_log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    print('\nEvaluation only')
    test_loss, test_acc = test(testloader, model, criterion, use_cuda)
    print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
    return

def test(testloader, model, criterion, use_cuda):
    landmarks = []
    shapes = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, batch_data in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = batch_data['image']
        targets = batch_data['landmarks']
        shape = batch_data['shapes']
        shapes.append(shape.cpu().data.numpy())

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        landmarks.append(outputs.cpu().data.numpy())

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        losses.update(loss.data, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} '.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()

    shapes = np.concatenate(shapes, axis=0)
    Hs = shapes[:,0]
    Ws = shapes[:,1]
    landmarks = np.concatenate(landmarks, axis = 0)
    landmarks = np.reshape(landmarks, (-1, 68, 2))
    landmarks = np.transpose(landmarks, (0, 2, 1))
    landmarks = np.reshape(landmarks, (-1, 136))

    angles = angleCal_py(landmarks, Hs, Ws)

    dataframe = pd.DataFrame(angles)
    dataframe.to_csv('pred_angles.csv',index=False)

    return (losses.avg, 0)


if __name__ == '__main__':
    main()