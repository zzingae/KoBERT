from model import make_model
from dataloader import QnADataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def get_accuracy_from_logits(logits, labels):
    probs = F.softmax(logits, dim=1)
    output = torch.argmax(probs, dim=1)
    #Convert probabilities to predictions, 1 being positive and 0 being negative
    #Check which predictions are the same as the ground truth and calculate the accuracy
    acc = (output == labels).float().mean()
    return acc

def train(model, criterion, opti, train_loader):

    max_eps = 2
    gpu = 'cuda'
    print_every = 10

    for ep in range(max_eps):
        for it, (question, attn_masks, answer) in enumerate(train_loader):
            #Clear gradients
            opti.zero_grad()  
            #Converting these to cuda tensors
            # question, attn_masks, answer = question.cuda(gpu), attn_masks.cuda(gpu), answer.cuda(gpu)

            #Obtaining the logits from the model
            tgt_mask = attn_masks
            output = model(question, attn_masks, answer, tgt_mask)
            logits = model.generator(output)

            #Computing loss
            loss = criterion(logits.squeeze(-1), answer.float())

            #Backpropagating the gradients
            loss.backward()

            #Optimization step
            opti.step()

            if (it + 1) % print_every == 0:
                acc = get_accuracy_from_logits(logits, answer)
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(it+1, ep+1, loss.item(), acc))


model, vocab = make_model(3)
criterion = nn.BCEWithLogitsLoss()
opti = optim.Adam(model.parameters(), lr = 2e-5)

# Creating instances of training and validation set
maxlen=25
path='./data/ChatBotData.csv'
train_set = QnADataset(path, vocab, maxlen)
# train_Q, eval_Q, train_A, eval_A = train_test_split(question, answer, test_size=0.33, random_state=42)
# Creating instances of training and validation dataloaders
num_workers=5
train_loader = DataLoader(train_set, batch_size = 64)
# val_loader = DataLoader(val_set, batch_size = 64, num_workers = 5)

train(model,criterion,opti,train_loader)
