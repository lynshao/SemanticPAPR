import argparse, os, pdb, sys, time, math
import torch
import torch.nn as nn
import torch.nn.init as init
import scipy.io as sio
import torchvision
import torchvision.transforms as transforms

def load_CIFAR10(BatchSize):
    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BatchSize, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1000, shuffle=False, num_workers=2)
    return trainloader, testloader

def get_RRC():
    # args.sps = 10, roll-off factor = 0.5
    RRCmat = sio.loadmat('RRC.mat')
    data = RRCmat['Pulse']
    RRC = torch.from_numpy(data).float()
    return RRC.unsqueeze(1), RRC.size(dim=-1)


def args_parser():
    parser = argparse.ArgumentParser(description='Main')
    # Sequence arguments
    parser.add_argument('--M', type=int, default = 128, help = '# total sucarriers')
    parser.add_argument('--N', type=int, default = 64, help = '# allocated sucarriers')
    parser.add_argument('--lenCP', type=int, default = 16, help = 'length of CP')
    parser.add_argument('--precoding', type=int, default = 1, help = 'DFT precoding or not')
    parser.add_argument('--mapping', type=int, default = 0, help = '0=>OFDMA, 1=>LFDMA, 2=>IFDMA')
    parser.add_argument('--sps', type=int, default = 10, help = 'sample per symbol')
    parser.add_argument('--lamb', type=float, default = 0, help = 'coefficient of PAPR loss')
    parser.add_argument('--thres', type=float, default = 0, help = 'threshold of PAPRdB')
    parser.add_argument('--hstd', type=float, default = 1, help = 'std of h')
    parser.add_argument('--fading', type=int, default = 0, help = 'fading or not')
    parser.add_argument('--clip', type=float, default = 0, help = 'clip ratio > 1; 0. means no clipping')

    parser.add_argument('--BatchSize', type=int, default= 256)
    parser.add_argument('--snr', type=float, default= 10.)

    parser.add_argument('--lr', type=float, default= 1e-3, help = "learning rate")
    parser.add_argument('--wd', type=float, default= 0.005, help = "weight decay")
    parser.add_argument('--adamW', type=int, default=1, help="use adamW or not")
    parser.add_argument('--numepoch', type=int, default=100, help="total number of epochs")
    parser.add_argument('--load', type=int, default= 0, help = 'load trained model')

    args = parser.parse_args()
    return args



_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 25.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
