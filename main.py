import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch, time, pdb, os, random, math
import matplotlib.pyplot as plt

from utils import load_CIFAR10, get_RRC, args_parser, progress_bar
from resnet import ResNetTx, ResNetRx
from scipy.io import savemat

# set seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class SemanticComm(nn.Module):
    def __init__(self, args):
        super(SemanticComm, self).__init__()
        self.args = args
        self.M = args.M
        self.N = args.N
        self.Enc = ResNetTx()
        self.Dec = ResNetRx()
        # load RRC pulse
        self.RRC, self.lenRRC = get_RRC()
        # generate carrier
        fc = 25e6; # carrier frequency
        fb = 10e6; # baseband frequency
        fs = fb * self.args.sps; # sampling rate
        numSamples = int(256/self.N) * (self.M+self.args.lenCP) * self.args.sps + self.lenRRC * 2 # number of samples 
        tt = torch.arange(0, numSamples/fs, 1/fs)
        self.carrier_cos = np.sqrt(2) * torch.cos(2 * np.pi * fc * tt).to(self.args.device)
        self.carrier_sin = np.sqrt(2) * torch.sin(2 * np.pi * fc * tt).to(self.args.device)

    def power_norm(self, feature):
        in_shape = feature.shape
        sig_in = feature.reshape(in_shape[0], -1)
        # each pkt in a batch, compute the mean and var
        sig_mean = torch.mean(sig_in, dim = 1)
        sig_std = torch.std(sig_in, dim = 1)
        # normalize
        sig_out = (sig_in-sig_mean.unsqueeze(dim=1))/(sig_std.unsqueeze(dim=1)+1e-8)
        return sig_out.reshape(in_shape)

    def pulse_filter_complex_sig(self, x):
        xreal = F.conv1d(x[:,:,0].unsqueeze(1), self.RRC.flip(dims=[2]).to(self.args.device), stride=1, padding=self.lenRRC-1)
        ximag = F.conv1d(x[:,:,1].unsqueeze(1), self.RRC.flip(dims=[2]).to(self.args.device), stride=1, padding=self.lenRRC-1)
        # ---------------------------------------------------------- convert back to complex
        x = torch.cat([xreal.squeeze(1).unsqueeze(2),ximag.squeeze(1).unsqueeze(2)], dim=2)
        return torch.view_as_complex(x)

    def compute_PAPR(self, x_t, len_data_t):
        # truncation, only compute the PAPR of the signal part (keep the signal length to len_data_t)
        truncloc = torch.arange(int((self.lenRRC+1)/2),int(((self.lenRRC+1)/2+len_data_t*self.args.sps)))
        x_t = torch.index_select(x_t, 1, truncloc.to(self.args.device))
        # ---------------------------------------------------------- compute PAPR
        data_t_power = torch.square(torch.abs(x_t))
        meanPower = torch.mean(data_t_power, dim = 1)
        maxPower = torch.max(data_t_power, dim = 1).values
        PAPRdB = 10 * torch.log10(maxPower/meanPower)
        PAPRloss = F.relu(PAPRdB-self.args.thres).mean()
        return PAPRdB, PAPRloss

    def channel(self, data_x):
        inputBS, len_data_x = data_x.size(0), data_x.size(1)
        noise_std = 10 ** (-self.args.snr * 1.0 / 10 / 2)
        # real channel
        AWGN = torch.normal(0, std=noise_std, size=(inputBS, len_data_x), requires_grad=False).to(self.args.device)

        if self.args.fading == 1:
            # at the receiver, the equivalent noise power is reduced by self.hh**2
            self.hh = np.random.rayleigh(self.args.hstd, 1) # sigma and number of samples
        else:
            self.hh = np.array([1])

        data_r = torch.from_numpy(self.hh).type(torch.float32).to(self.args.device) * data_x + AWGN
        return data_r

    def clip(self, x):
        x_power_mean_amp = torch.sqrt(torch.mean(torch.square(x), dim = 1))
        thres = self.args.clip * x_power_mean_amp
        non_neg_diff = F.relu(torch.abs(x) - thres.unsqueeze(1))
        x = (1 - non_neg_diff/(torch.abs(x)+1e-8)) * x # scale the symbol with amplitude larger than thres
        return x

    def forward(self, x):
        inputBS = x.shape[0]
        # =================================================================================== Encoding
        # the dim of enc output is fixed to [BS, 512] (rate = 1/12)
        x = self.Enc(x)
        # power norm
        x = self.power_norm(x)
        # reshape to a vector BS*256*2 (for construting complex symbols)
        x = x.view(inputBS, 256, 2)
        # (512 real symbols) to (256 complex symbols); power of real = 1; power of complex = 2
        x = torch.view_as_complex(x)
        # =================================================================================== modulation
        # total subcarriers = M = 128, # allocated subcarriers = N = 64
        numOFDM = int(256/self.N)
        # ---------------------------------------------------------- modulation
        # oneOFDM = x[:, (idx*self.N):((idx+1)*self.N)] # take out each OFDM symbols separately
        x = x.view(inputBS, numOFDM, self.N)
        # DFT precoding, if necessary
        if self.args.precoding == 1:
            x = torch.fft.fft(x, dim=-1, norm="ortho")
            x = torch.fft.fftshift(x, dim = -1)
        # ------------------------------------------------ mapping to subcarriers
        # determine locations of subcarriers
        if self.args.mapping == 0:
            # OFDMA, random mapping
            loc = torch.randperm(self.M)[:self.N]
        elif self.args.mapping == 1:
            # LFDMA, localized mapping
            startloc = torch.randint(self.M-self.N+1, size=())
            loc = torch.arange(startloc, startloc+self.N)
        elif self.args.mapping == 2:
            # IFDMA, interleaved mapping
            startloc = torch.randint(int(self.M/self.N), size=())
            loc = torch.arange(startloc, self.M, 2)
        loc = loc.to(self.args.device)
        # power of x = 2; power of data_f = 1
        data_f = torch.zeros(inputBS, numOFDM, self.M, dtype = torch.complex64).to(self.args.device)
        data_f[:, :, loc] = x
        # IFFT; power of data_f = 1, power of data_t = 1
        data_t = torch.fft.ifft(data_f, n=None, dim=-1, norm="ortho")
        # ---------------------------------------------------------- add CP
        len_data_t = numOFDM * (self.M+self.args.lenCP)
        if self.args.lenCP != 0:
            data_t = torch.cat([data_t[:,:,-self.args.lenCP:], data_t], dim=-1)
        # reshape back to a packet
        data_t = data_t.view(inputBS, len_data_t)

        # ---------------------------------------------------------- oversampling
        data_t_over = torch.zeros(inputBS, len_data_t*self.args.sps, dtype = torch.complex64).to(self.args.device)
        data_t_over[:, np.arange(0, len_data_t*self.args.sps, self.args.sps)] = data_t
        # ---------------------------------------------------------- pulse shaping (real in, complex out)
        data_t_over = torch.view_as_real(data_t_over)
        data_x = self.pulse_filter_complex_sig(data_t_over)
        # ---------------------------------------------------------- RF signal (real)
        data_x = data_x.real * self.carrier_cos[:data_x.size(1)] - data_x.imag * self.carrier_sin[:data_x.size(1)]
        # ---------------------------------------------------------- clipping
        if self.args.clip != 0.0:
            data_x = self.clip(data_x)
        # ---------------------------------------------------------- compute PAPR
        PAPRdB, PAPRloss = self.compute_PAPR(data_x, len_data_t)

        # =================================================================================== Channel
        data_r = self.channel(data_x)

        # =================================================================================== demodulation
        # ---------------------------------------------------------- baseband signal
        data_r_real = data_r * self.carrier_cos[:data_r.size(1)]
        data_r_imag = data_r * -self.carrier_sin[:data_r.size(1)]
        data_r_cpx = torch.cat([data_r_real.unsqueeze(2),data_r_imag.unsqueeze(2)], dim=2)

        # ---------------------------------------------------------- matched filtering (real in, complex out)
        data_r_filtered = self.pulse_filter_complex_sig(data_r_cpx)

        # synchronization and samling
        samplingloc = torch.arange(self.lenRRC-1, data_r_filtered.size(1)-self.lenRRC, self.args.sps)
        y = torch.index_select(data_r_filtered, 1, samplingloc.to(self.args.device))
  
        # remove CP
        y = y.view(inputBS, numOFDM, self.M+self.args.lenCP)
        y = y[:,:,self.args.lenCP:]
        # FFT
        y = torch.fft.fft(y, n=None, dim=-1, norm="ortho")
        # demapping
        y = torch.index_select(y, 2, loc)
        # IDFT, if necessary
        if self.args.precoding == 1:
            y = torch.fft.fftshift(y, dim = -1)
            y = torch.fft.ifft(y, dim=-1, norm="ortho")
        # reshape to a packet, BS*4*64 -> BS*256
        y = y.view(inputBS, numOFDM * self.N)
        # complex to real, BS*256*2
        y = torch.view_as_real(y)
        # reshape to a compressed image
        y = y.view(y.size(0), 8, 8, 8)
        y = self.Dec(y)

        return y, PAPRdB, PAPRloss
        


def train(epoch, args, model, trainloader, best_PSNR):
    # ============================================= training
    print('\nEpoch: %d' % epoch)
    model.train()
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(args.device)
        args.optimizer.zero_grad()
        outputs, PAPRdB, PAPRloss = model(inputs)
        if args.lamb == 0.0:
            loss = args.loss(outputs, inputs)
        else:
            loss = args.loss(outputs, inputs) + args.lamb * PAPRloss
        loss.backward()
        args.optimizer.step()
        progress_bar(batch_idx, len(trainloader), 'bestPSNR: %.2f, MSE: %.4f, PAPRdB: %.4f'%(best_PSNR,loss,PAPRdB.mean()))

def test(epoch, args, model, testloader, best_PSNR, saveflag = 1):
    model.eval()
    psnr_all_list = []
    MSEnoAvg = nn.MSELoss(reduction = 'none')
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader):
            b,c,h,w=inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]
            inputs = inputs.to(args.device)
            outputs, PAPRdB, _ = model(inputs)
            loss = MSEnoAvg(outputs, inputs)
            MSE_each_image = (torch.sum(loss.view(b,-1),dim=1))/(c*h*w)
            PSNR_each_image = 10 * torch.log10(1 / MSE_each_image)
            one_batch_PSNR = PSNR_each_image.data.cpu().numpy()
            psnr_all_list.extend(one_batch_PSNR)
            if batch_idx == 0:
                PAPRdBarray = PAPRdB
            else:
                PAPRdBarray = torch.cat([PAPRdBarray, PAPRdB],dim=0)
        test_PSNR=np.mean(psnr_all_list)
        test_PSNR=np.around(test_PSNR,5)
        
        print("test_PSNR, = meanPAPR", (test_PSNR,PAPRdBarray.mean().cpu().numpy()))

    if saveflag == 1:
        # Save checkpoint.
        if test_PSNR > best_PSNR:
            print('Saving..')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'test_PSNR': test_PSNR,
                'PAPR': PAPRdBarray,
            }

            filename = "snr" + str(int(args.snr)) + "_precoding" + str(args.precoding) + "_mapping" + str(args.mapping)+ "_lamb" + str(args.lamb) + "_clip" + str(args.clip) + "_fading" + str(args.fading) + "_hstd" + str(args.hstd)
            # save checkpoint
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/'+ filename + '.pth')
            best_PSNR = test_PSNR


            # save matlab file
            mdic = {"PSNR": best_PSNR, "PAPRarray": PAPRdBarray.cpu().numpy()}
            savemat("./checkpoint/" + filename + ".mat", mdic)

        return best_PSNR


def main(model, args):
    # ======================================================= load datasets
    trainloader, testloader = load_CIFAR10(args.BatchSize)

    # ======================================================= start (train or test)
    if args.load == 0:
        # start training
        best_PSNR = 0
        for epoch in np.arange(args.numepoch):
            train(epoch, args, model, trainloader, best_PSNR)
            best_PSNR = test(epoch, args, model, testloader, best_PSNR)

            args.scheduler.step()
    else:
        # load a trained model and test
        filename = "snr" + str(int(args.snr)) + "_precoding" + str(args.precoding) + "_mapping" + str(args.mapping)+ "_lamb" + str(args.lamb) + "_clip" + str(0.0) + "_fading" + str(args.fading) + "_hstd" + str(args.hstd)
        checkpoint = torch.load("./checkpoint/" + filename + ".pth")
        model.load_state_dict(checkpoint['model'])
        print("=======>>>>>>>> Successfully load the pretrained data!")
        test(0, args, model, testloader, 0, saveflag = 1)
        # pdb.set_trace()



if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.loss = nn.MSELoss()
    # ======================================================= Initialize the model
    model = SemanticComm(args).to(args.device)
    if args.device == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    # ======================================================= Optimizer
    if args.adamW == 1:
        args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
    else:
        args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # ======================================================= lr scheduling
    lambdafn = lambda epoch: (1-epoch/args.numepoch)
    args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambdafn)

    main(model, args)
