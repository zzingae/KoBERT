import torch
from model import make_model
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


model_path = './output/step_10.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_decoder_layers = 3
model, vocab = make_model(num_decoder_layers)
model.load_state_dict(torch.load(model_path, map_location=device)['model'])
model.eval()

sp  = SentencepieceTokenizer(get_tokenizer())

print('insert question:')
question = input()
max_len = 20

tokens = sp(question)
tokens = ['[CLS]'] + tokens + ['[SEP]']
token_ids = [vocab.token_to_idx[tok] for tok in tokens]
# unsqueeze(0) for Batch position (zzingae)
token_ids = torch.tensor(token_ids).unsqueeze(0)
attn_mask = (token_ids != vocab.token_to_idx['[PAD]']).long()

answer = model.greedy_decode(token_ids, attn_mask, max_len, vocab)

print('question: {}'.format(sp(question)))
print('answer: '+''.join([vocab.idx_to_token[idx] for idx in answer[0,1:]]))