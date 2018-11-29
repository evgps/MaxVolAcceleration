import shutil
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def save_checkpoint(state, is_best, save_dir = '.'):
    """
    Save the training self.model
    """
    filename = "{}/checkpoint.pth.tar".format(save_dir)
    torch.save(state, filename)
    
    if is_best:
        shutil.copyfile(filename, '{}/model_best.pth.tar'.format(save_dir))
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.values = []

    def update(self, val, n=1):
        self.values += [val]
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def get_var(self):
        return np.var(self.values)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    
def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    return F.cross_entropy(outputs, labels)


def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.alpha
    T = params.temperature

    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                         F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
          F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


# def accuracy(outputs, labels):
#     """
#     Compute the accuracy, given the outputs and labels for all images.
#     Args:
#         outputs: (np.ndarray) output of the model
#         labels: (np.ndarray) [0, 1, ..., num_classes-1]
#     Returns: (float) accuracy in [0,1]
#     """
#     outputs = np.argmax(outputs, axis=1)
#     return np.sum(outputs==labels)/float(labels.size)    
    
    
def adjust_learning_rate(optimizer, epoch, lr0 = None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

        
    #lr = lr0 * (0.5 ** (epoch // 30))
    lr = lr0 * (0.99**(epoch//30))    

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def validate(val_loader, model, print_info=None, device='cuda', is_svhn = False):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # switch to evaluate mode
    model.to(device)
    model.eval()

    with torch.no_grad():
        
        for i, (input, target) in enumerate(val_loader):
            
            input = input.to(device)
            target = target.to(device)
            if is_svhn:
                target = target%10
            # compute output
            end = time.time()
            output = model(input)
            batch_time.update(1000.*(time.time() - end))

            loss = loss_fn(output,target)

            
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            end = time.time()
            
            if i % 50 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
                
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))



        if print_info is None:
            print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
        else:
            with open(print_info['filename'], 'a') as file:
                file.write('{!s}\t{!s}\t{!s}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{!s}\n'.format(
                    print_info['dataset'],
                    print_info['arch'],
                    print_info['compression_rate'],
                    print_info['batch_size'],
                    losses.avg,
                    losses.get_var(),
                    top1.avg,
                    top5.avg,
                    batch_time.avg,
                    batch_time.get_var(),
                    device))
                file.close()

    return top1, top5, losses

    
def train(train_loader, model, teacher_model, optimizer, epoch, kd_params = None, device='cuda', is_svhn=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.to(device)
    model.train()
          
    if kd_params is not None:
        teacher_model.to(device)
        teacher_model.eval()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
         
        # measure data loading time
        data_time.update(time.time() - end)
        
        input = input.to(device)
        
        # input = input.to(device)
        target = target.to(device)
        if is_svhn:
            target = target%10
        # compute output
        output = model(input)
        
        if kd_params is not None:
            # teacher_output = teacher_model(input)
            # loss = loss_fn_kd(output, target, teacher_output, kd_params)# + loss_fn(output,target,is_svhn)
            loss = loss_fn(output,target)
        else:
            loss = loss_fn(output,target)
            


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 20 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time, data_time=data_time,
                      loss=losses,top1=top1, top5=top5))
