import torch
from torch.utils.data import Dataset
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
import pandas as pd


def padding_tokens(tokens, maxlen):
    #Preprocessing the text to be suitable for BERT
    tokens = ['[CLS]'] + tokens + ['[SEP]'] #Insering the CLS and SEP token in the beginning and end of the sentence
    if len(tokens) < maxlen:
        return tokens + ['[PAD]' for _ in range(maxlen - len(tokens))] #Padding sentences
    else:
        return tokens[:maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length


class QnADataset(Dataset):
    def __init__(self, filename, vocab, maxlen, use_emotion):

        #Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, header=0, encoding='utf-8')
        # self.df = pd.read_csv(filename, delimiter = '\t')
        self.sp = SentencepieceTokenizer(get_tokenizer())
        self.vocab = vocab
        self.maxlen = maxlen
        self.use_emotion = use_emotion

        self.sp.tokens.index('!')
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        question = self.df.loc[index, 'Q']
        answer = self.df.loc[index, 'A']
        emotion = self.df.loc[index, 'label']

        if self.use_emotion=='True':
            # 일상다반서 0, 이별(부정) 1, 사랑(긍정) 2로 레이블링
            if emotion==0:
                emotion_word = '일상 '
            elif emotion==1: # self.sp('이별') --> ['▁이', '별']
                emotion_word = '부정 '
            else:
                emotion_word = '사랑 '
            qtoks = self.sp(emotion_word + question)
        else:
            qtoks = self.sp(question)

        atoks = self.sp(answer)

        qtoks = ['[CLS]'] + qtoks
        # qtoks = padding_tokens(qtoks, self.maxlen)
        if len(qtoks) < self.maxlen:
            qtoks += ['[PAD]']*(self.maxlen - len(qtoks))
        else:
            qtoks = qtoks[:self.maxlen]

        # tgt=['▁차', '근', '차', '근', '▁계획을', '▁세워', '보', '세요', '.', '[SEP]', '[PAD]']
        # inp=['[SEP]', '▁차', '근', '차', '근', '▁계획을', '▁세워', '보', '세요', '.', '[SEP]']
        # 마지막의 [PAD] 부분은 loss 계산에서 제외됨.
        atoks = ['[SEP]'] + atoks
        # atoks = padding_tokens(atoks, self.maxlen+1)
        if len(atoks) < self.maxlen:
            atoks += ['[SEP]']
            atoks += ['[PAD]']*(self.maxlen+1 - len(atoks))
        else:
            atoks = atoks[:self.maxlen] + ['[SEP]']

        qids = [self.vocab.token_to_idx[t] for t in qtoks]
        qids_tensor = torch.tensor(qids) #Converting the list to a pytorch tensor

        aids = [self.vocab.token_to_idx[t] for t in atoks]
        aids_tensor = torch.tensor(aids) #Converting the list to a pytorch tensor

        return qids_tensor, aids_tensor