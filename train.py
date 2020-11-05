import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
from model import make_model
from dataloader import QnADataset
from utils import *
# from nltk.translate.bleu_score import sentence_bleu


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
    max_eps = 500
    print_every = 100
    save_every = 5000

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

            logits = model(question, attn_masks, tgt_input, tgt_mask)

            #Computing loss
            loss = criterion(logits, tgt_output)

            #Backpropagating the gradients
            loss.backward()

            #Optimization step
            opti.step()

            if (step + 1) % print_every == 0:
                acc = get_accuracy(logits, tgt_output, vocab.token_to_idx['[PAD]'])
                avg_loss = loss.item() / question.shape[0]
                write_summary(writer, {'loss': avg_loss, 'acc': acc, 'lr': opti._rate}, step+1)

                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(step+1, epoch+1, avg_loss, acc))
                print('Q: '+''.join([vocab.idx_to_token[idx] for idx in question[0]]))
                print('logits A: '+''.join([vocab.idx_to_token[idx] for idx in torch.argmax(logits, dim=2)[0]]))
                print('target A: '+''.join([vocab.idx_to_token[idx] for idx in tgt_output[0]]))

            if (step + 1) % save_every == 0:
                avg_loss, acc = evaluation(device, model, vocab, val_loader, criterion)
                save_ckpt(model, opti, step+1, epoch+1, save_path)
                write_summary(writer, {'loss': avg_loss, 'acc': acc}, step+1)
                print("Evaluation {} complete. Loss : {} Accuracy : {}".format(step+1, avg_loss, acc))

            step += 1
            
    save_ckpt(model, opti, step+1, epoch+1, save_path)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_decoder_layers = 3
    maxlen=25
    path='./data/ChatBotData.csv'
    save_path='./output'
    label_smoothing = 0.4

    # training data: 8377
    # steps for one epoch: 130
    train_portion = 0.7
    train_batch_size = 64

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model, vocab = make_model(num_decoder_layers)
    model.to(device)

    criterion = LabelSmoothing(len(vocab), vocab.token_to_idx['[PAD]'], smoothing=label_smoothing)
    # opti = get_std_opt(model)
    opti = get_my_opt(model,learning_rate=1,warmup_steps=4000)

    dataset = QnADataset(path, vocab, maxlen)
    train_val_ratio = [int(len(dataset)*train_portion)+1, int(len(dataset)*(1-train_portion))]
    train_set, val_set = random_split(dataset, train_val_ratio)
    # Creating instances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size = train_batch_size, shuffle=True, num_workers=5)
    val_loader = DataLoader(val_set, batch_size = 500, num_workers = 5)

    train_val(device, model, vocab, train_loader, val_loader, criterion, opti, save_path)


if __name__ == "__main__":
    main()
