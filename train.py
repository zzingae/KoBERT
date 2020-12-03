import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
from model import make_model
from dataloader import QnADataset
from utils import *
# from nltk.translate.bleu_score import sentence_bleu
import argparse
import random


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

def evaluation(device, model, vocab, val_loader, criterion, args):

    # model.eval() will notify all your layers that you are in eval mode, 
    # that way, batchnorm or dropout layers will work in eval mode instead of training mode.
    model.eval()
    # torch.no_grad() impacts the autograd engine and deactivate it. 
    # It will reduce memory usage and speed up computations but you won’t be able to backprop.
    avg_loss = 0
    avg_acc = 0
    with torch.no_grad():
        for _, (sources, targets) in enumerate(val_loader):
            batch =  Batch(sources, targets, vocab.token_to_idx['[PAD]'])

            if not device.type=='cpu':
                batch.src, batch.src_mask = batch.src.cuda(device), batch.src_mask.cuda(device)
                batch.trg, batch.trg_mask = batch.trg.cuda(device), batch.trg_mask.cuda(device)
                batch.trg_y = batch.trg_y.cuda(device)

            #Obtaining the log_prob after log_softmax (zzingae)
            logits = model(batch.src, batch.src_mask, batch.trg, batch.trg_mask, 
                           vocab, args.maxlen, use_teacher_forcing=False)

            #accumulate loss and accuracy (zzingae)
            avg_loss += (criterion(logits, batch.trg_y) / batch.ntokens) * batch.src.shape[0]
            avg_acc += get_accuracy(logits, batch.trg_y, vocab.token_to_idx['[PAD]']) * batch.src.shape[0]

    model.train()
    return avg_loss/len(val_loader.dataset), avg_acc/len(val_loader.dataset)

def train_val(device, model, vocab, train_loader, val_loader, criterion, opti, save_path, args):

    step=0
    print_every = 10
    writer = SummaryWriter(save_path)

    for epoch in range(args.max_epochs):

        for _, (sources, targets) in enumerate(train_loader):
            batch =  Batch(sources, targets, vocab.token_to_idx['[PAD]'])

            # Clear gradients
            opti.optimizer.zero_grad()  

            if not device.type=='cpu':
                batch.src, batch.src_mask = batch.src.cuda(device), batch.src_mask.cuda(device)
                batch.trg, batch.trg_mask = batch.trg.cuda(device), batch.trg_mask.cuda(device)
                batch.trg_y = batch.trg_y.cuda(device)

            use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False

            #Computing loss
            logits = model(batch.src, batch.src_mask, batch.trg, batch.trg_mask, 
                           vocab, args.maxlen, use_teacher_forcing)
            loss = criterion(logits, batch.trg_y) / batch.ntokens
            #Backpropagating the gradients
            loss.backward()

            #Optimization step
            opti.step()

            if (step + 1) % print_every == 0:
                acc = get_accuracy(logits, batch.trg_y, vocab.token_to_idx['[PAD]'])
                write_summary(writer, {'loss': loss, 'acc': acc, 'lr': opti._rate}, step+1)

                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(step+1, epoch+1, loss, acc))
                print('Q: '+''.join([vocab.idx_to_token[idx] for idx in batch.src[0]]))
                print('teacher forced? : {}'.format(use_teacher_forcing))
                print('logits A: '+''.join([vocab.idx_to_token[idx] for idx in torch.argmax(logits, dim=2)[0]]))
                print('target A: '+''.join([vocab.idx_to_token[idx] for idx in batch.trg_y[0]]))

            if (step + 1) % args.save_every == 0:
                avg_loss, acc = evaluation(device, model, vocab, val_loader, criterion, args)
                save_ckpt(model, opti, step+1, epoch+1, save_path)
                write_summary(writer, {'loss': avg_loss, 'acc': acc}, step+1)
                print("Evaluation {} complete. Loss : {} Accuracy : {}".format(step+1, avg_loss, acc))

            step += 1
            
    save_ckpt(model, opti, step+1, epoch+1, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./data/ChatBotData.csv')
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--maxlen', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=2500)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=1000) # due to small number of training data, number of epochs set to be large.
    parser.add_argument('--warmup_steps', type=int, default=4000) # due to small number of training data, number of epochs set to be large.
    
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('--label_smoothing', type=float, default=0.4)
    parser.add_argument('--train_portion', type=float, default=0.7) # training data: 8377 if 0.7
    parser.add_argument('--learning_rate', type=float, default=1.0)
    parser.add_argument('--save_every', type=int, default=500)

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    if not os.path.exists('./output'):
        os.mkdir('./output')

    model, vocab = make_model(args.num_decoder_layers)
    # opti = get_std_opt(model)
    opti = get_my_opt(model,learning_rate=args.learning_rate,warmup_steps=args.warmup_steps)
    model = nn.DataParallel(model.to(device))
    criterion = LabelSmoothing(len(vocab), vocab.token_to_idx['[PAD]'], smoothing=args.label_smoothing)

    dataset = QnADataset(args.data_path, vocab, args.maxlen)
    train_val_ratio = [int(len(dataset)*args.train_portion)+1, int(len(dataset)*(1-args.train_portion))]
    train_set, val_set = random_split(dataset, train_val_ratio)
    # Creating instances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(val_set, batch_size = args.batch_size, num_workers = args.num_workers)

    train_val(device, model, vocab, train_loader, val_loader, criterion, opti, save_path='./output', args=args)
