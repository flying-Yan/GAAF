
import torch
import torch.nn as nn

import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime

from torchvision.utils import save_image
import torchvision.transforms as transforms





def Unary_op(x, t):
    x = x.float()
    if t == 0:
        y = x
    elif t == 1:
        y = torch.zeros_like(x)
    elif t == 2:
        y = torch.pow(x,2)
    elif t == 3:
        y = torch.pow(x,3)
    elif t == 4:
        y = torch.sign(x)*torch.sqrt(torch.abs(x) + 1e-8)
    elif t == 5:
        y = torch.log(torch.abs(x) + 1e-8)
    elif t == 6:
        y = torch.sin(x)
    elif t == 7:
        y = torch.cos(x)
    elif t == 8:
        y = 1.0/(1+torch.exp(-x))
    elif t == 9:
        y = torch.tan(x)
    elif t == 10:
        y = torch.atan(x)
    elif t == 11:
        y = torch.erf(x)
    elif t == 12:
        y = torch.erfc(x)
    elif t == 13:
        y = torch.exp(-torch.abs(x) + 1e-8)
    elif t == 14: 
        y = torch.exp(-x*x)
    elif t == 15:
        y = torch.max(x,torch.zeros_like(x))
    elif t == 16:
        y = torch.min(x,torch.zeros_like(x))
    elif t == 17:
        y = torch.abs(x)
    else:
        y = -x

    return y

def Binary_op(x, y, t):
    if t == 0:
        z = x + y   
    elif t == 1:
        z = x - y
    elif t == 2:
        z = x*y
    elif t == 3:
        z = x/(y+1e-8)
    elif t == 4:
        z = x/(x+y+1e-8)
    elif t == 5:
        z = torch.max(x,y)
    elif t == 6:
        z = torch.min(x,y)
    elif t == 7:
        z = x/(1+torch.exp(-y))
    elif t == 8:
        z = torch.exp(-torch.pow(x-y,2))
    else:
        z = torch.exp(-torch.abs(x-y))


    return z


def Binarize(tensor,quant_mode='det'):
        
    return torch.sign(tensor + 1e-6)


class BB_b3(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BB_b3, self).__init__(*kargs, **kwargs)

    
    def forward(self, input):
        
        input.data = Binarize(input.data)
        
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

   
class LearnableBias(nn.Module):
    def __init__(self, channel):
        super(LearnableBias,self).__init__()
        
        
        self.s_1 = nn.Parameter(torch.ones(1, channel,1,1), requires_grad = True)
        self.s_2 = nn.Parameter(torch.zeros(1, channel,1,1), requires_grad = True)
        
        self.s_3 = nn.Parameter(torch.ones(1, channel,1,1), requires_grad = True)
        self.s_4 = nn.Parameter(torch.zeros(1, channel,1,1), requires_grad = True)
        
        self.sb = nn.Parameter(torch.ones(1, channel,1,1), requires_grad = True)

    def forward(self,x, UB_ops):
        
        if UB_ops[0] == 19:
            x_1 = self.s_1.expand_as(x)*x
        elif UB_ops[0] == 20:
            x_1 = x + self.s_2.expand_as(x)
        elif UB_ops[0] == 21:
            x_1 = self.s_1.expand_as(x)
        else:
            x_1 = Unary_op(x, UB_ops[0])

        if UB_ops[1] == 19:
            x_2 = self.s_3.expand_as(x)*x
        elif UB_ops[1] == 20:
            x_2 = x + self.s_4.expand_as(x)
        elif UB_ops[1] == 21:
            x_2 = self.s_3.expand_as(x)
        else:
            x_2 = Unary_op(x, UB_ops[1])
        
        if UB_ops[2] == 10:
            out = self.sb.expand_as(x)*x_1 + (1-self.sb).expand_as(x)*x_2
        else:
            out = Binary_op(x_1,x_2, UB_ops[2])

        
        return out


class Res18_Net(nn.Module):
    #### ResnetE18

    def __init__(self, UB_ops):
        super(Res18_Net, self).__init__()
        
        self.tanh = nn.Hardtanh(-1.3,1.3)
               
        self.UB_ops = UB_ops


        self.conv1 = nn.Sequential(
                nn.BatchNorm2d(3, affine = False),
                nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(64),
            )

        # block 1
        self.b_1_1_B1 = BB_b3(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.b_1_1_N1 = nn.BatchNorm2d(64)
        self.b_1_1_B2 = BB_b3(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.b_1_1_N2 = nn.BatchNorm2d(64)
        
        self.b_1_2_B1 = BB_b3(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.b_1_2_N1 = nn.BatchNorm2d(64)
        self.b_1_2_B2 = BB_b3(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.b_1_2_N2 = nn.BatchNorm2d(64)
        

        # block 2
        self.b_2_1_B1 = BB_b3(64, 128, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.b_2_1_N1 = nn.BatchNorm2d(128)
        self.b_2_1_B2 = BB_b3(128, 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.b_2_1_N2 = nn.BatchNorm2d(128)

        self.short_2_B = BB_b3(64, 128, kernel_size = 1, stride = 2, padding = 0, bias = False)
        self.short_2_N = nn.BatchNorm2d(128)
        
        
        self.b_2_2_B1 = BB_b3(128, 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.b_2_2_N1 = nn.BatchNorm2d(128)
        self.b_2_2_B2 = BB_b3(128, 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.b_2_2_N2 = nn.BatchNorm2d(128)

        
        # block 3
        self.b_3_1_B1 = BB_b3(128, 256, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.b_3_1_N1 = nn.BatchNorm2d(256)
        self.b_3_1_B2 = BB_b3(256, 256, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.b_3_1_N2 = nn.BatchNorm2d(256)

        self.short_3_B = BB_b3(128, 256, kernel_size = 1, stride = 2, padding = 0, bias = False)
        self.short_3_N = nn.BatchNorm2d(256)

        
        self.b_3_2_B1 = BB_b3(256, 256, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.b_3_2_N1 = nn.BatchNorm2d(256)
        self.b_3_2_B2 = BB_b3(256, 256, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.b_3_2_N2 = nn.BatchNorm2d(256)

        
        # block 4
        self.b_4_1_B1 = BB_b3(256, 512, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.b_4_1_N1 = nn.BatchNorm2d(512)
        self.b_4_1_B2 = BB_b3(512, 512, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.b_4_1_N2 = nn.BatchNorm2d(512)

        self.short_4_B = BB_b3(256, 512, kernel_size = 1, stride = 2, padding = 0, bias = False)
        self.short_4_N = nn.BatchNorm2d(512)

        
        self.b_4_2_B1 = BB_b3(512, 512, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.b_4_2_N1 = nn.BatchNorm2d(512)
        self.b_4_2_B2 = BB_b3(512, 512, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.b_4_2_N2 = nn.BatchNorm2d(512)

        self.bn_1_1 = LearnableBias(64)
        self.bn_1_2 = LearnableBias(64)
        self.bn_1_3 = LearnableBias(64)
        self.bn_1_4 = LearnableBias(64)
        
        self.bn_2_1 = LearnableBias(64)
        self.bn_2_2 = LearnableBias(128)
        self.bn_2_3 = LearnableBias(64)
        self.bn_2_4 = LearnableBias(128)
        self.bn_2_5 = LearnableBias(128)

        self.bn_3_1 = LearnableBias(128)
        self.bn_3_2 = LearnableBias(256)
        self.bn_3_3 = LearnableBias(128)
        self.bn_3_4 = LearnableBias(256)
        self.bn_3_5 = LearnableBias(256)
        
        self.bn_4_1 = LearnableBias(256)
        self.bn_4_2 = LearnableBias(512)
        self.bn_4_3 = LearnableBias(256)
        self.bn_4_4 = LearnableBias(512)
        self.bn_4_5 = LearnableBias(512)

        
        self.fc = nn.Sequential(
                nn.Linear(512,10)
            )

        self.avg = nn.Sequential(
                nn.ReLU(),
                nn.AvgPool2d(4,4),
            )
        
        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            80: {'lr': 1e-3},
            150: {'lr': 5e-4},
            200: {'lr': 1e-4},
            240: {'lr': 5e-5},
            270: {'lr': 1e-5}
        }

                
    def forward(self, x):
        
        x = self.conv1(x)   # 64 56x56
        
        
        y = self.bn_1_1(x, self.UB_ops)
        y = self.tanh(y)    # block 1
        y = self.b_1_1_B1(y)
        y = self.b_1_1_N1(y)
        y = self.bn_1_2(y, self.UB_ops)
        y = self.tanh(y)
        y = self.b_1_1_B2(y)
        y = self.b_1_1_N2(y)

        x = x + y

        y = self.bn_1_3(x, self.UB_ops)
        y = self.tanh(y)
        y = self.b_1_2_B1(y)
        y = self.b_1_2_N1(y)
        y = self.bn_1_4(y, self.UB_ops) 
        y = self.tanh(y)
        y = self.b_1_2_B2(y)
        y = self.b_1_2_N2(y)

        x = x + y

        y = self.bn_2_1(x, self.UB_ops)
        y = self.tanh(y)      # block 2
        y = self.b_2_1_B1(y)
        y = self.b_2_1_N1(y)
        y = self.bn_2_2(y, self.UB_ops) 
        y = self.tanh(y)
        y = self.b_2_1_B2(y)
        y = self.b_2_1_N2(y)

        x = self.bn_2_3(x, self.UB_ops)
        x = self.tanh(x)
        x = self.short_2_B(x)
        x = self.short_2_N(x)

        x = x + y

        y = self.bn_2_4(x, self.UB_ops) 
        y = self.tanh(y)      
        y = self.b_2_2_B1(y)
        y = self.b_2_2_N1(y)
        y = self.bn_2_5(y, self.UB_ops) 
        y = self.tanh(y)
        y = self.b_2_2_B2(y)
        y = self.b_2_2_N2(y)

        x = x + y

        y = self.bn_3_1(x, self.UB_ops) 
        y = self.tanh(y)      # block 3
        y = self.b_3_1_B1(y)
        y = self.b_3_1_N1(y)
        y = self.bn_3_2(y, self.UB_ops) 
        y = self.tanh(y)
        y = self.b_3_1_B2(y)
        y = self.b_3_1_N2(y)
        
        x = self.bn_3_3(x, self.UB_ops)
        x = self.tanh(x)
        x = self.short_3_B(x)
        x = self.short_3_N(x)

        x = x + y

        y = self.bn_3_4(x, self.UB_ops) 
        y = self.tanh(y)      
        y = self.b_3_2_B1(y)
        y = self.b_3_2_N1(y)
        y = self.bn_3_5(y, self.UB_ops) 
        y = self.tanh(y)
        y = self.b_3_2_B2(y)
        y = self.b_3_2_N2(y)

        x = x + y

        y = self.bn_4_1(x, self.UB_ops) 
        y = self.tanh(y)      # block 4
        y = self.b_4_1_B1(y)
        y = self.b_4_1_N1(y)
        y = self.bn_4_2(y, self.UB_ops)
        y = self.tanh(y)
        y = self.b_4_1_B2(y)
        y = self.b_4_1_N2(y)

        x = self.bn_4_3(x, self.UB_ops)
        x = self.tanh(x)
        x = self.short_4_B(x)
        x = self.short_4_N(x)

        x = x + y

        y = self.bn_4_4(x, self.UB_ops) 
        y = self.tanh(y)      
        y = self.b_4_2_B1(y)
        y = self.b_4_2_N1(y)
        y = self.bn_4_5(y, self.UB_ops)
        y = self.tanh(y)
        y = self.b_4_2_B2(y)
        y = self.b_4_2_N2(y)

        x = x + y
   
        x = self.avg(x)

        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x
     
    
def fitness_score(gpu,seed, UBs, save_p, epochs, batch_size):  
    # 
    print(UBs)
    best_prec1 = 0
        
    save_path = os.path.join(save_p)

    
    torch.cuda.set_device(gpu)

   
    model = Res18_Net(UBs)

   
    # Data loading code
    default_transform = {
        'train': get_transform('cifar10',
                               input_size=32, augment=True),
        'eval': get_transform('cifar10',
                              input_size=32, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    regime = getattr(model, 'regime', {0: {'optimizer': 'Adam',
                                           'lr': 0.01,
                                           'momentum': 0.9,
                                           'weight_decay': 0}})
    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.cuda()
    model.cuda()

    val_data = get_dataset('cifar10', 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True)



    train_data = get_dataset('cifar10', 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    for epoch in range(0, epochs):
        optimizer = adjust_optimizer(optimizer, epoch, regime)

        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer)


        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, epoch)

        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \t'
                     'best_prec {best_prec1:.3f} \n'
                     .format(epoch + 1,  train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5, best_prec1 = best_prec1))


        if best_prec1 < 11:
            print('-- fitness_score stoped early -----')
            return best_prec1
        
    return best_prec1


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    #if args.gpus and len(args.gpus) > 1:
        #model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (inputs, target) in enumerate(data_loader):
        target = target.cuda()
        input_var = Variable(inputs.cuda(), volatile=not training)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1.3,1.3))


        
    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)

   
