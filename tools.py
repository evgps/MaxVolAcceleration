import numpy as np
import torch
assert torch.__version__ >= '0.4.0'
import matplotlib.pylab as plt

from numpy.linalg import svd
from randomized_svd import torch_randomized_svd
from feedforwardcompressed import slice_blocks
from IPython import display
import copy
import os
import pickle
import time
import tables
import h5py


def save_V_array(V_array, filename='./models/V.pt'):
    torch.save(V_array, filename)

def load_V_array(filename='./models/V.pt'):
    return list(map(lambda V: nn.Parameter(V), torch.load(filename)))

def save_inv_ind(model, filename='./models/inv_ind.pt'):
    torch.save([model.inv_V_array, model.ind_array], filename)
    
def load_inv_ind(filename='./models/inv_ind.pt'):
    return torch.load(filename)


def plot_sigma(Sigma, title=''):
    plt.yscale('log')
    plt.xlabel('dimensionality')
    plt.ylabel('singular values')
    plt.title(title)
    if isinstance(Sigma, np.ndarray):
        plt.plot(Sigma / Sigma[0])
    else:
        plt.plot(Sigma.cpu().detach().numpy() / Sigma[0].cpu().detach().numpy())


def plot_sigma(Sigma, title='', file=None):
    plt.yscale('log')
    plt.xlabel('dimensionality')
    plt.ylabel('singular values')
    plt.title(title)
    if isinstance(Sigma, np.ndarray):
        plt.semilogy(Sigma / Sigma[0])
    else:
        plt.semilogy(Sigma.cpu().detach().numpy() / Sigma[0].cpu().detach().numpy())
    if file is not None:
        plt.savefig(file)



def compute_approx(U, Sigma, V, r):
    X_approx = U[:, :r].mm(torch.diag(Sigma[:r])).mm(V[:, :r].t())
    return X_approx


def get_pseudoinverse(A):
    return torch.inverse(A.t().mm(A)).mm(A.t())

def propagate_through_layer(layer, X, batch_size=2048):
    '''
    This function is needed cause GPU memory can be overflowed 
    by the data on some middle layers
    '''
    loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        s = layer(X[0:1].detach().cuda()).shape
        shape = [X.shape[0]] + list(s[1:])
        X_output = torch.empty(shape, requires_grad=False, device='cpu')
        for idx, batch in enumerate(loader):
            res = layer(batch.cuda()).cpu()
            X_output[idx * batch_size : (idx + 1) * batch_size] = layer(batch.cuda()).cpu()
    return X_output


def square_norm(x):
    return torch.sum(x * x)

def train_epoch(model, train_loader, optimizer, epoch, alpha, train_loss=[], train_loss_original=[]):
    device = 'cuda'
    model.train()
    criterion = nn.CrossEntropyLoss().cuda()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model.forward_old(data)
        activation_idx = 0
        # Compute additional loss:
        add_loss = 0.0
        activation_idx = 0
        batch = data
        for layer in model.features:
            batch = layer(batch)
            if 'ReLU' in str(layer):
                V = model.V_array[activation_idx].cuda()
                mat_batch = batch.view(batch.shape[0], -1).t()
                add_loss += square_norm(mat_batch - V @ (V.t() @ mat_batch))
                activation_idx += 1
        add_loss /= batch.shape[0]
        # Compute loss:
        loss = criterion(output, target)
        train_loss_original.append(loss.item())
        loss += alpha * add_loss
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if len(train_loss) % 10 == 0:
            display_losses(train_loss, train_loss_original)
            
def train_modified_loss(model, train_loader, optimizer, alpha, train_loss=[], 
                        train_loss_original=[], n_epochs=10):
    display.clear_output(wait=True)
    for epoch in range(1, n_epochs + 1):
        train_epoch(model, train_loader, optimizer, epoch, 
                    alpha=alpha, train_loss=train_loss, train_loss_original=train_loss_original)


def display_losses(loss, loss_original, xlabel='#iteration', ylabel='loss', title='Training loss'):
    display.clear_output(wait=True)
    # log y axis
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.semilogy(loss, c='g')
    plt.title('Modified loss')
    #plt.grid(True)
    # log y axis
    plt.subplot(122)
    plt.semilogy(loss_original, c='r')
    plt.title('Original loss')
    #plt.grid(True)
    plt.show()

class HaveInterlayers(torch.nn.Module):
    def __init__(self, model, dataloader, device='cuda'):
        super(HaveInterlayers, self).__init__()
        _, blocks = slice_blocks(model, dataloader)
        self.block_model = []
        for start, end in blocks:
            self.block_model.append(model.features[start:end])
        self.features = model.features
        self.classifier = model.classifier
        self.forward = model.forward
        
    def get_interlayer(self, x):
        interlayers = []
        for blocks in self.block_model:
            x = blocks(x)
            interlayers.append(x.cpu().detach().numpy().reshape(x.shape[0],-1))
        return interlayers
    



#Function that calc and save randomised_SVD(max_r, k) for each interlayer from root_path/arch_name 
def dump_svd(arch_name, root_path, max_r=10000, k=50):
    if not os.path.exists(root_path+arch_name+'/svd/'):
        os.mkdir(root_path+arch_name+'/svd/')
    if len(os.listdir(root_path+arch_name+'/svd/')):
        print('Directory {} is not empty, skipping calculation.\n'.format(root_path+arch_name+'/svd/'))
    else:
        # Function for loading interlayers, dumped as h5 files
        def load_interlayer(root_path, arch_name):
            dir_path = root_path + arch_name
            if not os.path.exists(dir_path):
                print('No such files in dir {}'.format(dir_path))
                return
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            X = []
            block_nums = []

            for file in sorted(files, key=lambda x: int(x.split('.')[0])):
                block_nums += [file.split('.')[0]]
                filename = dir_path+"/"+file
                if not os.path.exists(filename):
                    print('Wrong parsing?')
                    break
                #load one first array from '.hd5'
                with h5py.File(filename, mode='r') as f:
                    for key,val in f.items():
                        X.append(val[()])
                        break
            return X, block_nums

        arch_path = root_path+arch_name+'/'
        svd_path = arch_path+'svd/'
        assert os.path.exists(arch_path)
        if not os.path.exists(svd_path):
            os.mkdir(svd_path)
        print('Loading interlayers...\n')
        X, block_nums = load_interlayer(root_path, arch_name)
        def load_and_svd(args):
            X, block_num = args
            X = X.reshape(X.shape[0],-1)
            #maintain tall matrixes
            sample_size = min(X.shape[0], X.shape[1])
            X = X[:sample_size,:]
            filename = svd_path+"SVD_after_block_{}.pickle".format(block_num)
            if os.path.exists(filename):
                print('Already calculated')
                return
            
            start = time.time()
            U,S,V = torch_randomized_svd(torch.DoubleTensor(X))
            print('Calculated SVD for {:.2f} seconds'.format(time.time()-start))
            
            with open(filename, 'wb') as f:
                pickle.dump((U.cpu(),S.cpu(),V.cpu()), f, protocol=4)
            return
        
        for i in map(load_and_svd, zip(X, block_nums)):
            print('Start calculating SVD for new block')
        


def create_folders(config):
    if not os.path.exists(config.storage_path+"interlayers/"):
        os.mkdir(config.storage_path+"interlayers/")
    if not os.path.exists(config.storage_path+"pretrained/"):
        os.mkdir(config.storage_path+"pretrained/")
    if not os.path.exists(config.storage_path+"compressed/"):
        os.mkdir(config.storage_path+"compressed/")
    if not os.path.exists(config.storage_path+"results/"):
        os.mkdir(config.storage_path+"results/")

    if not os.path.exists(config.storage_path+"interlayers/"+config.dataset):
        os.mkdir(config.storage_path+"interlayers/"+config.dataset)
    if not os.path.exists(config.storage_path+"pretrained/"+config.dataset):
        os.mkdir(config.storage_path+"pretrained/"+config.dataset)
    if not os.path.exists(config.storage_path+"compressed/"+config.dataset):
        os.mkdir(config.storage_path+"compressed/"+config.dataset)
    if not os.path.exists(config.storage_path+"results/"+config.dataset):
        os.mkdir(config.storage_path+"results/"+config.dataset)

    if not os.path.exists(config.storage_path+"interlayers/"+config.dataset+'/'+config.model):
        os.mkdir(config.storage_path+"interlayers/"+config.dataset+'/'+config.model)
    if not os.path.exists(config.storage_path+"pretrained/"+config.dataset+'/'+config.model):
        os.mkdir(config.storage_path+"pretrained/"+config.dataset+'/'+config.model)
    if not os.path.exists(config.storage_path+"compressed/"+config.dataset+'/'+config.model):
        os.mkdir(config.storage_path+"compressed/"+config.dataset+'/'+config.model)
    if not os.path.exists(config.storage_path+"results/"+config.dataset+'/'+config.model):
        os.mkdir(config.storage_path+"results/"+config.dataset+'/'+config.model)