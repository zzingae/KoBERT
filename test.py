import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model

input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
model, vocab  = get_pytorch_kobert_model()
sequence_output, pooled_output = model(input_ids, input_mask, token_type_ids)
print(sequence_output)
print(pooled_output)

from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
tok_path = get_tokenizer()
sp  = SentencepieceTokenizer(tok_path)
print(sp('한국어 모델을 공유합니다.'))

