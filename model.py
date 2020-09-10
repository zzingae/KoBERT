import torch
import torch.nn as nn
from transformers import BertModel
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformer import *


def make_model(N=6, d_model=768, d_ff=1024, h=8, dropout=0.1):
    # To copy Bert embedding layer, d_model should be the same for Generator.
    # Since d_model=768 in decoder is too big to train, d_ff is set from 3072 to 1024. (zzingae)
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    bert, vocab = get_pytorch_kobert_model()
    vocab_size = len(vocab)

    model = Chatbot(bert,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, vocab_size), c(position)),
        nn.Sequential(Embeddings(d_model, vocab_size), c(position)),
        Generator(d_model, vocab_size))
    
    return model, vocab

class Chatbot(nn.Module):
    # borrowed from:
    # https://medium.com/swlh/painless-fine-tuning-of-bert-in-pytorch-b91c14912caa 

    def __init__(self, bert, decoder, src_embed, tgt_embed, generator, freeze_bert=True):
        super(Chatbot, self).__init__()
        #Instantiating BERT model object 
        self.bert = bert
        self.decoder = decoder
        # self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
        # share weights between tgt_embed and bert embeddding (zzingae)
        # only copy weights from bert embedding to generator (zzingae)
        self.tgt_embed[0].lut.weight = self.bert.embeddings.word_embeddings.weight
        self.generator.proj.weight = copy.deepcopy(self.bert.embeddings.word_embeddings.weight)
        # tgt_embed and bert embed will be frozen while generator won't be (zzingae)
        self.tgt_embed[0].lut.weight.requires_grad = False

        #Freeze bert layer
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        # only decoder params are randomly initialized and trained (zzingae)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, seq, attn_masks, tgt, tgt_mask):
        #Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert(seq, attention_mask = attn_masks)

        output = self.decoder(self.tgt_embed(tgt), cont_reps, attn_masks, tgt_mask)
        return self.generator(output)