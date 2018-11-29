import numpy as np
import torch
from copy import deepcopy
from sklearn.utils.extmath import randomized_svd
from maxvolpy.maxvol import rect_maxvol
from torch import nn
from torch.nn import ReLU, MaxPool2d
import torch.utils.data

def get_batch(dataloader):
    batch = dataloader.__iter__().__next__()
    if isinstance(batch, list):
        batch = batch[0]
    return batch

def propagate(dataloader, layers, device='cuda'):
    layers = layers.to(device)
    with torch.no_grad():
        batch = get_batch(dataloader).to(device)
        s = layers(batch).shape
        # n_samples = min(len(dataloader.dataset), np.prod(s[1:]))
        n_samples = len(dataloader.dataset)
        X_new = torch.zeros(n_samples, s[1], s[2], s[3], device='cpu')
        idx = 0
        for batch in dataloader:
            if isinstance(batch, list):
                batch = batch[0]
            bs = batch.shape[0]
            X_new[idx : idx + bs] = layers(batch.to(device))
            idx += dataloader.batch_size
            if idx >= n_samples:
                break
    return X_new

def get_W_sample_dot_W_conv_matrix(conv, indices, input_shape, device='cpu'):
    '''
    Computes W_sample @ W_conv matrix 
    '''
    conv.to(device)
    bias = deepcopy(conv.bias)
    conv.bias = None
    c1, h1, w1 = input_shape
    m = c1 * h1 * w1
    _, c2, h2, w2 = conv(torch.zeros(1, c1, h1, w1, device=device)).shape
    n = c2 * h2 * w2
    dataloader = torch.utils.data.DataLoader(torch.eye(m).reshape(m, c1, h1, w1), 
                                     batch_size=256, shuffle=False)
    W = torch.ones(indices.numel(), m)
    idx = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(i)
            batch = batch.reshape(-1, c1, h1, w1).to(device)
            bs = batch.shape[0]
            W[:, idx : idx + bs] = conv(batch).reshape(bs, -1)[:, indices].t()
            idx += bs
        if bias is not None:
            conv.bias = nn.Parameter(bias)
    return W

def get_W_bn_and_b_bn(bn, bias, W_sc, indices, h, w, device='cuda'):
    channels = indices // (h * w)
    den = torch.sqrt(bn.state_dict()['running_var'] + bn.eps)
    sd = bn.state_dict()
    if 'weight' in sd:
        bn_alpha = sd['weight'] / den
        bn_bias = sd['bias'] - sd['running_mean'] * bn_alpha
    else:
        bn_alpha = 1 / den
        bn_bias = - sd['running_mean'] * bn_alpha
    bn_alpha = bn_alpha[channels].unsqueeze(0)
    W = W_sc * bn_alpha.t()
    new_bias = bn_alpha * bias + bn_bias[channels]
    return W, new_bias


def slice_blocks(model, dataloader, device='cuda'):
    '''
    Determines CNN blocks' boundaries and calculate the shapes of transition matrices between CNN's blocks
    Note:
        all activations are supposed to be [Leaky]ReLU;
        block slicing is made for convolutional part of the model (it should be called model.features)

    Args:
        model: a pre-trained model
        dataloader: a torch.utils.data.DataLoader object
    Returns:
        transition_matrix_shapes (list of tuples): shapes of transition matrices,
        block_boundaries (list of tuples): for each bloak corresponds (start_layer_idx, end_layer_idx + 1)
    '''
    model = model.to(device)
    model.eval()

    x = dataloader.__iter__().__next__()[0][:1]
    x = x.to(device)

    block_boundaries = []
    layer_shapes = [x.numel()]
    block_shapes = []

    # why you put it to device again?
    for layer in model.features:
        x = layer(x)
        layer_shapes += [x.numel()]

    maxpool_flag = False
    for idx, layer in list(enumerate(model.features))[::-1]:
        s = str(layer)
        if 'Dropout' in s:
            continue
        if 'MaxPool' in s:
            maxpool_flag = True
            block_boundaries += [idx + 1]
            block_shapes += [layer_shapes[idx + 1]]
        elif 'ReLU' in s:
            if not maxpool_flag:
                block_boundaries += [idx + 1]
                block_shapes += [layer_shapes[idx]]
            maxpool_flag = False
        else:
            maxpool_flag = False

    block_shapes += [layer_shapes[0]]
    block_boundaries += [0]

    transition_matrix_shapes = list(zip(block_shapes[::-1][:-1], block_shapes[::-1][1:]))
    block_boundaries = list(zip(
        block_boundaries[::-1][:-1],
        block_boundaries[::-1][1:]
    ))
    return transition_matrix_shapes, block_boundaries

def slice_blocks_student(model, device = None):
    layer_names = np.array([l._get_name() for l in list(dict(model.features.named_children()).values())])

    blocks_boundaries = []
    idx_end = len(layer_names)
    prev_end = None
    for idx, name in list(enumerate(layer_names))[::-1]:
        if name == 'Sequential':
            idx_start = idx
            blocks_boundaries.append((idx_start, idx_end))
            idx_end = idx_start
        elif name not in ['ReLU', 'Flatten']:
            continue
        elif name == 'ReLU':
            idx_start = idx + 1
            blocks_boundaries.append((idx_start, idx_end))
            idx_end = idx_start
        elif name == "Flatten":
            idx_start = idx
            blocks_boundaries.append((idx_start, idx_end))
            idx_end = idx_start    
    
    return blocks_boundaries[::-1]

class FeedforwardCompressed(torch.nn.Module):
    def __init__(self, model, dataloader, compression_rate, device='cuda'):
        super(FeedforwardCompressed, self).__init__()
        self.device = device
        # Check compression rate:
        for i in range(len(compression_rate) - 2):
            if list(map(lambda x: bool(x), compression_rate[i:i+3])) == [False, True, False]:
                raise ValueError('Compression rate array should not ' + 
                    'contain [None, Some_number, None] subsequences!')
                
        W_sizes, block_indices = slice_blocks(model, dataloader, device=device)
        self.W_sizes = W_sizes
        self.mp = torch.nn.MaxPool2d(1 + 1)
        
        # Save blocks:
        self.blocks = []
        for (begin, end) in block_indices:
            self.blocks += [deepcopy(model.features[begin:end]).to(device)]
        self.classifier = deepcopy(model.classifier).to(device)
            
        # Compute rank list:
        self.r = []
        for ws, cr in zip(W_sizes, compression_rate):
            if cr is None:
                self.r += [None]
            elif isinstance(cr, float):
                self.r += [int(cr * ws[1])]
            elif isinstance(cr, int):
                self.r += [min(cr, ws[1])]
                
        # Find blocks with maxpool:
        self.has_maxpool = []
        for block in self.blocks:
            self.has_maxpool += [False]
            for layer in block:
                if 'MaxPool' in str(layer):
                    self.has_maxpool[-1] = True
        
        # Recompute mode:
        if self.r[0] is None:
            self.mode = [None]
        else:
            self.mode = ['first']
        for prev_r, r, next_r in zip(self.r[:-2], self.r[1:-1], self.r[2:]):
            if r is None:
                self.mode += [None]
            else:
                if prev_r is None:
                    self.mode += ['first']
                elif next_r is None:
                    self.mode += ['last']
                else:
                    self.mode += ['middle']
        if self.r[-1] is None:
            self.mode += [None]
        else:
            self.mode += ['last']
            
        # Create additional lists:
        self.inv_V = []
        self.ind = []
        self.V = []
        self.shapes = []
        self.W = []
        self.biases = []
        self.sigmas = []
        # Remove later:
        self.old_ind = []
        return
    
    def _append_nones(self):
        self.inv_V += [None]
        self.ind += [None]
        self.V += [None]
        self.shapes += [None]
        self.W += [None]
        self.biases += [None]
        self.sigmas += [None]
        
        def _debugging_print(self):
            def len_or_none(array):
                if not array:
                    return '-1'
                elif array[-1] is None:
                    return None
                elif isinstance(array[-1], tuple):
                    return array[-1]
                else:
                    return array[-1].shape
            print(list(map(len_or_none, [self.inv_V, self.ind, 
                self.V, self.shapes, self.W])), flush=True)
            return
        
    def _compute_SVD(self, X, rank):
        '''
        Args:
            X: a 4-dimensional tensor (batch_size, channels, h, w)
        Returns:
            (U, Sigma, V^T) matrices
        '''
        return randomized_svd(X.reshape(X.shape[0], -1).numpy(),
            n_components=rank)
    
    def sample_bias(self, bias, ind, h, w):
        if bias is None:
            new_bias = torch.zeros(1, ind.numel())
        else:
            bias_idx = ind // (h * w)
            new_bias = bias[bias_idx].unsqueeze(0)
        return new_bias

    def _compute_maxpool_indices(self, indices, input_shape):
        c, h, w = input_shape
        output_shape = (c, h // 2, w // 2)
        def cij2p(c,i,j,input_shape):
            return c*input_shape[1]*input_shape[2] + i*input_shape[2] + j

        def p2cij(p,input_shape):
            c = p // (input_shape[1]*input_shape[2])
            i = (p-c*input_shape[1]*input_shape[2]) // (input_shape[2])
            j = p-c*input_shape[1]*input_shape[2]-i*input_shape[2]
            return c,i,j

        new_indices = np.zeros(indices.shape[0]*4)
        for n, p in enumerate(indices):
            c,i,j = p2cij(p,output_shape)
            new_indices[4*n:4*(n+1)] = [cij2p(c,2*i,2*j, input_shape), cij2p(c,2*i+1,2*j, input_shape),\
                                    cij2p(c,2*i,2*j+1, input_shape), cij2p(c,2*i+1,2*j+1, input_shape)]
        return torch.as_tensor(new_indices, dtype=torch.long)
    
    def fit(self, dataloader, SVD_list=None, verbose=False):
        # Output tensor:
        n_blocks = len(self.blocks)
        prev_V = None
        for idx, (r, mode, block, has_maxpool, w_shape) in enumerate(zip(self.r, self.mode, \
                                                self.blocks, self.has_maxpool, self.W_sizes)):
            if verbose:
                self._debugging_print()
            self._append_nones()
            X_new = propagate(dataloader, block, device='cuda')
            if mode is None:
                pass
            else:
                # SVD and MaxVol
                # in numpy:
                if SVD_list is None:
                    _, Sigma, V = self._compute_SVD(X_new, rank=r)
                    V = V.T
                else:
                    _, Sigma, V = SVD_list[idx]
                    Sigma = Sigma[:r]
                    V = V[:, :r]
                self.sigmas[-1] = Sigma
                ind, _ = rect_maxvol(V, maxK = max(13000, r))
                ind.sort()
                inv_V = np.linalg.pinv(V[ind, :])
                
                # For any mode except None:
                self.ind[-1] = torch.tensor(ind, dtype=torch.long)
                self.inv_V[-1] = torch.tensor(inv_V, dtype=torch.float32, device='cpu')
                V = torch.tensor(V, dtype=torch.float32, device='cpu')
                # We need to project back only in 'last' mode:
                if mode is 'first':
                    prev_V = V
                if mode is 'last':
                    self.V[-1] = nn.Parameter(V @ self.inv_V[-1])
                    self.shapes[-1] = X_new.shape[1:]
                # Compute W:
                if mode is not None:
                    bn = None
                    # Search for a convolution:
                    for layer in block:
                        if 'Conv' in str(layer):
                            conv = layer.to(self.device)
                        if 'Norm' in str(layer):
                            bn = layer.to(self.device)
                    # For maxpool: increase indices number:
                    if has_maxpool:
                        maxpool_input_shape = conv(
                            get_batch(dataloader)[0:1].to(self.device)).shape[1:]
                        # Remove later:
                        self.old_ind += [deepcopy(self.ind[-1])]
                        self.ind[-1] = self._compute_maxpool_indices(self.ind[-1], 
                                                                maxpool_input_shape)
                    # Compute input shape:
                    conv_input_shape = get_batch(dataloader).shape[1:]
                    W_sc = get_W_sample_dot_W_conv_matrix(conv, self.ind[-1], conv_input_shape,
                                                         device='cpu')
                    if mode is 'first':
                        self.W[-1] = W_sc.to(self.device)
                    else:
                        self.W[-1] = ((W_sc @ prev_V) @ self.inv_V[-2]).to(self.device)
                    # 
                    s = conv_input_shape
                    _, _, h, w = conv.to(self.device)(
                        torch.zeros(1, s[0], s[1], s[2], device=self.device)).shape
                    self.biases[-1] = self.sample_bias(conv.bias, self.ind[-1], h, w).to(self.device)
                    if bn is not None:
                        self.W[-1], self.biases[-1] = get_W_bn_and_b_bn(
                            bn, self.biases[-1], self.W[-1], self.ind[-1], h, w, device=self.device
                        )
                        self.W[-1].to(self.device)
                        self.biases[-1].to(self.device)
                    if self.has_maxpool[idx]:
                        self.biases[idx] = self.biases[idx][:, ::4]
                    prev_V = V
            #self.to(self.device)
            dataloader = torch.utils.data.DataLoader(X_new, shuffle=False, 
                                                     batch_size=dataloader.batch_size)
        return 
    
    def maxpool(self, x):
        batch_size = x.shape[0]
        return self.mp(x.reshape(batch_size, -1, 2)).squeeze(-1)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)
        
    def features(self, x):
        batch_size = x.shape[0]
        for idx, r in enumerate(self.r):
            if r is None:
                x = self.blocks[idx](x)
            else:
                if self.mode[idx] == 'first':
                    x = x.reshape(batch_size, -1)
                x = x @ self.W[idx].t()
                # If has maxpool:
                if self.has_maxpool[idx]:
                    x = self.maxpool(x)
                x += self.biases[idx]
                # General code for both 'middle' and 'last':
                x = torch.nn.functional.relu(x)
                if self.mode[idx] == 'last':
                    s = self.shapes[idx]
                    x = x @ self.V[idx].t()
                    x = x.reshape(batch_size, s[0], s[1], s[2])
        return x
    
    def to(self, *args, **kwargs):
        new_device = None
        for array in [self.V, self.inv_V, self.W, self.blocks, self.biases]:
            for idx, elem in enumerate(array):
                if elem is not None:
                    array[idx] = elem.to(*args, **kwargs)
                    if new_device is None:
                        try:
                            if array[idx].is_cuda:
                                new_device = array[idx].get_device()
                            else:
                                new_device = 'cpu'
                        except:
                            pass

        self.classifier = self.classifier.to(*args, **kwargs)
        self.device = new_device
        return
    
    def cuda(self, *args):
        new_device = None
        for array in [self.V, self.inv_V, self.W, self.blocks, self.biases]:
            for idx, elem in enumerate(array):
                if elem is not None:
                    array[idx] = elem.cuda(*args)
                    if new_device is None and isinstance(array[idx], torch.Tensor):
                        new_device = array[idx].get_device()
        self.classifier = self.classifier.cuda(*args)
        self.device = new_device
        return
    
    def cpu(self):
        self.to('cpu')
        


def block_speed(model, dataloader, device='cpu'):
    block_indx = slice_blocks_student(model, device=device)
    print(block_indx)
    times = []
    x = get_batch(dataloader)
    x = x.to(device)
    for start, end in block_indx:
        t = get_ipython().run_line_magic('timeit', '-n 300 -o model.features[start:end](x)')
        while t.stdev > t.average*0.1:
            print('Recalc')
            t = get_ipython().run_line_magic('timeit', '-n 300 -o model.features[start:end](x)')
        x = model.features[start:end](x)
        times.append(t)
    x = x.view(x.size(0), -1)
    t = get_ipython().run_line_magic('timeit', '-n 300 -o model.classifier(x)')
    while t.stdev > t.average*0.1:
        print('Recalc')
        t = get_ipython().run_line_magic('timeit', '-n 300 -o model.classifier(x)')
    times.append(t)        
    return times

def timeit_speed(model, dataloader, device):
    x = get_batch(dataloader)
    x = x.to(device)
    t = get_ipython().run_line_magic('timeit', '-n 300 -o model(x)')
    return [t]