import torch
from torch.utils.data import Dataset
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
import pandas as pd
from utils import subsequent_mask


def padding_tokens(tokens, maxlen):
    #Preprocessing the text to be suitable for BERT
    tokens = ['[CLS]'] + tokens + ['[SEP]'] #Insering the CLS and SEP token in the beginning and end of the sentence
    if len(tokens) < maxlen:
        return tokens + ['[PAD]' for _ in range(maxlen - len(tokens))] #Padding sentences
    else:
        return tokens[:maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length


class QnADataset(Dataset):
    def __init__(self, filename, vocab, maxlen):

        #Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, header=0, encoding='utf-8')
        # self.df = pd.read_csv(filename, delimiter = '\t')
        self.sp = SentencepieceTokenizer(get_tokenizer())
        self.vocab = vocab
        self.maxlen = maxlen

        self.sp.tokens.index('!')
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        question = self.df.loc[index, 'Q']
        answer = self.df.loc[index, 'A']
        emotion = self.df.loc[index, 'label']
        # 일상다반서 0, 이별(부정) 1, 사랑(긍정) 2로 레이블링
        if emotion==0:
            emotion_word = '일상 '
        elif emotion==1: # self.sp('이별') --> ['▁이', '별']
            emotion_word = '부정 '
        else:
            emotion_word = '사랑 '
        qtoks = self.sp(emotion_word + question)
        atoks = self.sp(answer)

        qtoks = padding_tokens(qtoks, self.maxlen)
        # +1 for including 'CLS' token which stands for starting token (zzingae)
        atoks = padding_tokens(atoks, self.maxlen+1)

        qids = [self.vocab.token_to_idx[t] for t in qtoks]
        qids_tensor = torch.tensor(qids) #Converting the list to a pytorch tensor

        aids = [self.vocab.token_to_idx[t] for t in atoks]
        aids_tensor = torch.tensor(aids) #Converting the list to a pytorch tensor

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (qids_tensor != self.vocab.token_to_idx['[PAD]']).long()
        tgt_mask = torch.squeeze(subsequent_mask(self.maxlen))

        # seq, attn_masks, tgt, tgt_mask
        return qids_tensor, attn_mask, aids_tensor, tgt_mask