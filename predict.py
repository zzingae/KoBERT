import torch
from model import make_model
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer

tmp_model, vocab = make_model(N=3)
sp  = SentencepieceTokenizer(get_tokenizer())
T = 12

sentence = '몇살이니?'
#Step 1: Tokenize
tokens = sp(sentence)
#Step 2: Add [CLS] and [SEP]
tokens = ['[CLS]'] + tokens + ['[SEP]']
#Step 3: Pad tokens
padded_tokens = tokens + ['[PAD]' for _ in range(T - len(tokens))]
attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
#Step 4: Segment ids
seg_ids = [0 for _ in range(len(padded_tokens))] #Optional!
#Step 5: Get BERT vocabulary index for each token
token_ids = [vocab.token_to_idx[tok] for tok in padded_tokens]
# token_ids = tokenizer.convert_tokens_to_ids(padded_tokens)

#Converting everything to torch tensors before feeding them to bert_model
token_ids = torch.tensor(token_ids).unsqueeze(0) #Shape : [1, 12]
attn_mask = torch.tensor(attn_mask).unsqueeze(0) #Shape : [1, 12]
seg_ids   = torch.tensor(seg_ids).unsqueeze(0) #Shape : [1, 12]

#Feed them to bert
hidden_reps, cls_head = bert_model(token_ids, attention_mask = attn_mask,\
                                  token_type_ids = seg_ids)
print(hidden_reps.shape)
#Out: torch.Size([1, 12, 768])
print(cls_head.shape)
#Out: torch.Size([1, 768])