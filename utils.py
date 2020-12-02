# copied from 
# http://nlp.seas.harvard.edu/2018/04/03/attention.html
# But slightly modified by zzingae

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


def get_std_opt(model):
    # std may mean standard (zzingae)
    # parameters with requires_grad=False will not be updated (zzingae)
    # Here, Adam's lr=0 is dummy value. Instead, lr from NoamOpt is used (zzingae)
    return NoamOpt(model.tgt_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def get_my_opt(model, learning_rate, warmup_steps):
    # customized optimization (zzingae)
    # parameters with requires_grad=False will not be updated (zzingae)
    # Here, Adam's lr=0 is dummy value. Instead, lr from NoamOpt is used (zzingae)
    return NoamOpt(model.tgt_embed[0].d_model, learning_rate, warmup_steps,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        # self.src_mask = (src != pad).unsqueeze(-2)
        # required src_mask shape for BERT: [batch_size,sequence_length] (zzingae)
        # required src_mask shape for decoder: [batch_size,1,sequence_length] (zzingae)
        self.src_mask = (src != pad)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        #  If the field size_average is set to False, the losses are instead summed for each minibatch.
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(2) == self.size
        true_dist = x.data.clone()
        # -2 may be from padding and target positions (zzingae)
        true_dist.fill_(self.smoothing / (self.size - 2))
        # put confidence to target positions (zzingae)
        true_dist.scatter_(2, target.data.unsqueeze(2), self.confidence)
        # model should not predict padding token (zzingae)
        true_dist[:, self.padding_idx] = 0
        # return padded indices in target (zzingae)
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            # put 0 for padded positions so that these padded positions return 0 loss (zzingae)
            true_dist[mask[:,0],mask[:,1],:] = 0.0
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))