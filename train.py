from model import make_model
from dataloader import QnADataset
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


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


def get_std_opt(model):
    # std may mean standard (zzingae)
    # parameters with requires_grad=False will not be updated (zzingae)
    return NoamOpt(model.tgt_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
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


def get_accuracy_from_logits(probs, labels):
    # probs = F.softmax(logits, dim=1)
    output = torch.argmax(probs, dim=2)
    #Convert probabilities to predictions, 1 being positive and 0 being negative
    #Check which predictions are the same as the ground truth and calculate the accuracy
    acc = (output == labels).float().mean()
    return acc

def train(model, train_loader, criterion, opti):

    max_eps = 100
    gpu = 'cuda'
    print_every = 100

    for ep in range(max_eps):
        for it, (question, attn_masks, answer, tgt_mask) in enumerate(train_loader):
            #Clear gradients
            # opti.zero_grad()  
            #Converting these to cuda tensors
            question, attn_masks, answer, tgt_mask = question.cuda(gpu), attn_masks.cuda(gpu), answer.cuda(gpu), tgt_mask.cuda(gpu)
            tgt_input = answer[:,:-1]
            tgt_output = answer[:,1:]

            output = model(question, attn_masks, tgt_input, tgt_mask)
            #Obtaining the log_prob after log_softmax (zzingae)
            log_prob = model.generator(output)

            #Computing loss
            loss = crit(Variable(log_prob), Variable(tgt_output))

            #Backpropagating the gradients
            # loss.backward()
            loss.requires_grad = True

            #Optimization step
            opti.step()

            if (it + 1) % print_every == 0:
                acc = get_accuracy_from_logits(torch.exp(log_prob), tgt_output)
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(it+1, ep+1, loss.item(), acc))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, vocab = make_model(3)
model.to(device)
# next(model.parameters()).device

crit = LabelSmoothing(len(vocab), vocab.token_to_idx['[PAD]'], smoothing=0.4)
opti = get_std_opt(model)
# Creating instances of training and validation set
maxlen=10
path='./data/ChatBotData.csv'
train_set = QnADataset(path, vocab, maxlen)
# train_Q, eval_Q, train_A, eval_A = train_test_split(question, answer, test_size=0.33, random_state=42)
# Creating instances of training and validation dataloaders
# , num_workers=5
train_loader = DataLoader(train_set, batch_size = 64)
# val_loader = DataLoader(val_set, batch_size = 64, num_workers = 5)

train(model, train_loader, crit, opti)


