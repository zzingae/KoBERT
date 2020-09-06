from model import make_model
from dataloader import QnADataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
# from nltk.translate.bleu_score import sentence_bleu


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


def get_accuracy(logits, labels, pad_id):
    output = torch.argmax(logits, dim=2)
    valid_pos = labels!=pad_id
    valid_num = valid_pos.sum().float()
    valid_sum = (valid_pos*(output==labels)).sum().float()
    return valid_sum / valid_num

def save_ckpt(model, opti, step, epoch, save_path):
    state = {
        'model': model.state_dict(),
        'optimizer': opti.optimizer.state_dict(),
        'step': step,
        'epoch': epoch
    }
    name = 'step_{}.pth'.format(step)
    torch.save(state, os.path.join(save_path,name))

def write_summary(writer, values, step):
    if 'lr' in values:
        name = 'train/'
        writer.add_scalar(name+"Learning_rate", values['lr'], step)
    else:
        name = 'eval/'

    writer.add_scalar(name+"Loss", values['loss'], step)
    writer.add_scalar(name+"Accuracy", values['acc'], step)

def evaluation(device, model, vocab, val_loader, criterion):

    # model.eval() will notify all your layers that you are in eval mode, 
    # that way, batchnorm or dropout layers will work in eval mode instead of training mode.
    model.eval()
    # torch.no_grad() impacts the autograd engine and deactivate it. 
    # It will reduce memory usage and speed up computations but you won’t be able to backprop.
    avg_loss = 0
    avg_acc = 0
    with torch.no_grad():
        for it, (question, attn_masks, answer, tgt_mask) in enumerate(val_loader):

            if not device.type=='cpu':
                question, attn_masks = question.cuda(device), attn_masks.cuda(device)
                answer, tgt_mask = answer.cuda(device), tgt_mask.cuda(device)
            tgt_input = answer[:,:-1]
            tgt_output = answer[:,1:]

            #Obtaining the log_prob after log_softmax (zzingae)
            logits = model(question, attn_masks, tgt_input, tgt_mask)

            #accumulate loss and accuracy (zzingae)
            avg_loss += criterion(logits, tgt_output)
            avg_acc += get_accuracy(logits, tgt_output, vocab.token_to_idx['[PAD]']) * question.shape[0]

    model.train()
    return avg_loss/len(val_loader.dataset), avg_acc/len(val_loader.dataset)

def train_val(device, model, vocab, train_loader, val_loader, criterion, opti, save_path):

    step=0
    max_eps = 1000
    print_every = 100
    save_every = 10000
    writer = SummaryWriter(save_path)

    for epoch in range(max_eps):

        for _, (question, attn_masks, answer, tgt_mask) in enumerate(train_loader):
            #Clear gradients
            opti.optimizer.zero_grad()  
            #Converting these to cuda tensors
            if not device.type=='cpu':
                question, attn_masks = question.cuda(device), attn_masks.cuda(device)
                answer, tgt_mask = answer.cuda(device), tgt_mask.cuda(device)
            tgt_input = answer[:,:-1]
            tgt_output = answer[:,1:]

            #Obtaining the log_prob after log_softmax (zzingae)
            logits = model(question, attn_masks, tgt_input, tgt_mask)

            #Computing loss
            loss = criterion(logits, tgt_output)

            #Backpropagating the gradients
            # loss.requires_grad = True
            loss.backward()

            #Optimization step
            opti.step()

            if (step + 1) % print_every == 0:
                acc = get_accuracy(logits, tgt_output, vocab.token_to_idx['[PAD]'])
                avg_loss = loss.item() / question.shape[0]
                write_summary(writer, {'loss': avg_loss, 'acc': acc, 'lr': opti._rate}, step+1)

                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(step+1, epoch+1, avg_loss, acc))
                print(torch.argmax(logits, dim=2)[0])
                print(tgt_output[0])

            if (step + 1) % save_every == 0:
                avg_loss, acc = evaluation(device, model, vocab, val_loader, criterion)
                save_ckpt(model, opti, step+1, epoch+1, save_path)
                write_summary(writer, {'loss': avg_loss, 'acc': acc}, step+1)
                print("Evaluation {} complete. Loss : {} Accuracy : {}".format(step+1, avg_loss, acc))

            step += 1


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_decoder_layers = 1
    maxlen=10
    path='./data/ChatBotData.csv'
    save_path='./output'
    train_portion = 0.7
    label_smoothing = 0.4
    train_batch_size = 64

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model, vocab = make_model(num_decoder_layers)
    model.to(device)

    criterion = LabelSmoothing(len(vocab), vocab.token_to_idx['[PAD]'], smoothing=label_smoothing)
    opti = get_std_opt(model)

    dataset = QnADataset(path, vocab, maxlen)
    train_val_ratio = [int(len(dataset)*train_portion)+1, int(len(dataset)*(1-train_portion))]
    train_set, val_set = random_split(dataset, train_val_ratio, 
                                      generator=torch.Generator().manual_seed(42))
    # Creating instances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size = train_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size = 1000)
    # val_loader = DataLoader(val_set, batch_size = 64, num_workers = 5)

    train_val(device, model, vocab, train_loader, val_loader, criterion, opti, save_path)


if __name__ == "__main__":
    main()
