import torch
from model import make_model
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from utils import greedy_decode


num_decoder_layers = 2
model_path = './outputs/output-N2/step_130000.pth'
model, vocab = make_model(num_decoder_layers)
model.load_state_dict(torch.load(model_path)['model'])
model.eval()

sp  = SentencepieceTokenizer(get_tokenizer())

question = '바보야'
max_len = 20

tokens = sp(question)
tokens = ['[CLS]'] + tokens + ['[SEP]']
token_ids = [vocab.token_to_idx[tok] for tok in tokens]
# unsqueeze(0) for Batch position (zzingae)
token_ids = torch.tensor(token_ids).unsqueeze(0)
# attention score: [Batch, Head, tgt_length, src_length] in src_attn (zzingae)
# unsqueeze(1) for tgt_length position (zzingae)
attn_mask = (token_ids != vocab.token_to_idx['[PAD]']).unsqueeze(1).long()

answer = greedy_decode(model, token_ids, attn_mask, max_len, vocab)

print('question: {}'.format(sp(question)))
print('answer: '+''.join([vocab.idx_to_token[idx] for idx in answer[0,1:]]))