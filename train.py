from model import make_model
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
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

def train(net, criterion, opti, train_loader, val_loader, args):

    for ep in range(args.max_eps):
        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            #Clear gradients
            opti.zero_grad()  
            #Converting these to cuda tensors
            seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)

            #Obtaining the logits from the model
            output = net(seq, attn_masks)
            logits = net.generator(output)

            #Computing loss
            loss = criterion(logits.squeeze(-1), labels.float())

            #Backpropagating the gradients
            loss.backward()

            #Optimization step
            opti.step()

            if (it + 1) % args.print_every == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(it+1, ep+1, loss.item(), acc))


# tmp_model, vocab = make_model(N=3)

tok_path = get_tokenizer()
sp  = SentencepieceTokenizer(tok_path)
print(sp('한국어 모델을 공유합니다.'))
print(vocab)

