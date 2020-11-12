# Transformer 기반 한국어 seq2seq Chatbot

<p align="center"> 
<img src="./imgs/architecture.png" alt="drawing" width="300"/> 
</p>

### Requirements

CUDA 10.1

### 학습 데이터

https://github.com/songys/Chatbot_data

### 학습 (fine-tuning)

KoBERT (encoder)를 freeze 후, Transformer decoder를 질문/답변 데이터셋으로 fine-tuning
- KoBERT의 학습된 input embedding으로 output embedding과 softmax weights를 초기화
- output embedding는 freeze 하고, 나머지 Transformer decoder의 weight를 학습

### 추론 및 attention visualization

<p align="center"> 
<img src="./imgs/attention.png" alt="drawing" width="600"/> 
</p>

### 참고

- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://medium.com/swlh/painless-fine-tuning-of-bert-in-pytorch-b91c14912caa 
