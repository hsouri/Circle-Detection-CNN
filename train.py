import argparse
import pandas as pd
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.optim as optim
import torch
import torch.nn as nn
from network import Net


parser = argparse.ArgumentParser(description='PyTorch Model Training')

parser.add_argument('--name',default='v8', type=str,
                    help='Name of the experiment.')
parser.add_argument('--out_file', default='new_out.txt',
                    help='path to output features file')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--resume',
                    default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data', default='train_set.csv', metavar='DIR',
                    help='path to imagelist file')
parser.add_argument('--print_freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--epochs', default=51, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number')
parser.add_argument('--save_freq', default=5, type=int,
                    help='Number of epochs to save after')


class NoisyImages(Dataset):
    def __init__(self, path_to_trainset, transform=None):
        self.dataset = pd.read_csv(path_to_trainset, sep=' ')
        self.transform = transform

    def __getitem__(self, idx):
        image = np.load(self.dataset.iloc[idx, 0].split(',')[0])
        target = [self.dataset.iloc[idx, 0].split(',')[i] for i in range(1, 4)]
        
        if self.transform:
            image = np.expand_dims(np.asarray(image), axis=0)
            image = torch.from_numpy(np.array(image, dtype=np.float32))
            target = torch.from_numpy(np.array(np.asarray(target), dtype=np.float32))
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.dataset)


def main():
    args = parser.parse_args()
    print(args)

    print("=> creating model")
    model = Net()

    if args.resume:
        print("=> loading checkpoint: " + args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
        args.start_epoch = int(args.resume.split('/')[1].split('_')[0])
        print("=> checkpoint loaded. epoch : " + str(args.start_epoch))

    else:
        print("=> Start from the scratch ")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)

    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), args.lr)

    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    trainset = NoisyImages(
        args.data,
        transforms.Compose([
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    output = open(args.out_file, "w")
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, criterion1, criterion2, optimizer, epoch, args, device, len(trainset), output)


def train(train_loader, model, criterion1, criterion2, optimizer, epoch, args, device, len, file):

    # switch to train mode
    model.train()
    running_loss = 0.0

    for i, (images, target) in enumerate(train_loader):

        images = images.to(device)
        target = target.to(device)

        output = model(images)

        loss = criterion1(output, target/200) + 0.1 * criterion2(output, target/200)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % args.print_freq == args.print_freq - 1 or i == int(len/args.batch_size):    # print every 50 mini-batches
            new_line = 'Epoch: [%d][%d/%d] loss: %f' % \
                       (epoch + 1, i + 1, int(len/args.batch_size) + 1, running_loss / args.print_freq)
            file.write(new_line + '\n')
            print(new_line)
            running_loss = 0.0

        if epoch % args.save_freq == 0:
            torch.save(model.module.state_dict(), 'saved_models/' + str(epoch) + '_epoch_' + args.name + '_checkpoint.pth.tar')


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate"""
    lr = args.lr
    if 20 < epoch <= 30:
        lr = 0.0001
    elif 30 < epoch :
        lr = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("learning rate -> {}\n".format(lr))


if __name__ == '__main__':
    main()
