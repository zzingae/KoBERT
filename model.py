import torch
import torch.nn as nn
from transformers import BertModel
from kobert.pytorch_kobert import get_pytorch_kobert_model

class SentimentClassifier(nn.Module):

    def __init__(self, freeze_bert = True):
        super(SentimentClassifier, self).__init__()
        #Instantiating BERT model object 

        self.bert_layer, self.vocab  = get_pytorch_kobert_model()
        # self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        
        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        #Classification layer
        self.cls_layer = nn.Linear(768, 1)

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        # input_ids=None,
        # attention_mask=None,
        # token_type_ids=None,

        # if token_type_ids is None:
        #     token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        #Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)

        #Obtaining the representation of [CLS] head
        cls_rep = cont_reps[:, 0]

        #Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)

        return logits