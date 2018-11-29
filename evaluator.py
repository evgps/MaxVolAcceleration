import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from maxvolpy.maxvol import maxvol

# CLASS THAT LOADING MODEL AND DO ALL TRAIN/TEST ROUTINE

class Evaluator():
    def __init__(self, model, loaders, workers=4, epochs=300, start_epoch=0, lr=0.05, momentum=0.9, weight_decay=5e-4,\
                print_freq=20, resume=False, evaluate=False, pretrained=False, half=False, save_dir='./data'):

        # super(VGGCifar, self).__init__()
        print('Load model:', model)
        self.model = model
        self.workers = workers
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.print_freq = print_freq
        self.resume = resume
        self.evaluate = evaluate
        self.pretrained = pretrained
        self.half = half
        self.save_dir = save_dir
        self.best_prec1 = 0

        # Check the save_dir exists or not
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.model.cuda()


        # optionally resume from a checkpoint
        if self.resume:
            if os.path.isfile(self.resume):
                print ("=> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.evaluate, checkpoint['epoch']))
            else:
                print ("=> no checkpoint found at '{}'".format(self.resume))


        cudnn.benchmark = True
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.train_loader = loaders['train']

        self.val_loader = loaders['val']

        # define loss function (self.criterion) and pptimizer
        self.criterion = nn.CrossEntropyLoss().cuda()


        if self.half:
            self.model.half()
            self.criterion.half()

        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)


    def trainval(self):

        if self.evaluate:
            self.validate(self.val_loader, self.model, self.criterion)
            return

        for epoch in range(self.start_epoch, self.epochs):
            self.adjust_learning_rate(epoch)

            # train for one epoch
            self.train(epoch)

            # evaluate on validation set
            prec1 = self.validate()

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.best_prec1
            self.best_prec1 = max(prec1, self.best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
            }, is_best, filename=os.path.join(self.save_dir, 'checkpoint_{}.tar'.format(epoch)))


    def train(self, epoch):
        """
            Run one train epoch
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (input, target) in enumerate(self.train_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target)
            if self.half:
                input_var = input_var.half()

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0].item()
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          epoch, i, len(self.train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1))


    def validate(self):
        """
        Run evaluation
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (inputs, targets) in enumerate(self.val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            output = self.model(inputs)
            loss = self.criterion(output, targets)

            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, targets)[0].item()
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(self.val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

        print(' * Prec@1 {top1.avg:.3f}'
              .format(top1=top1))

        return top1.avg



    def get_interlayer(self):
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (inputs, targets) in enumerate(self.val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            A = self.model.get_interlayer(inputs)
            yield A



    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
        self.lr = self.lr * (0.5 ** (epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training self.model
    """
    torch.save(state, filename)





class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
