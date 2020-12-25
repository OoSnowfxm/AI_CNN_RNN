import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
import torch.autograd as autograd

classes = {0:'B', 1:'I', 2:'O'}
nums_classes = 3
seq_len = 50

def getdataset(filename):
    f = open(filename,"r", encoding="utf-8")
    dataset = []
    row1 = []
    row2 = []
    for line in f:
        line1 = line.split("\t")
        if(line1 == ['\n']):
            dataset.append((row1, row2))
            row1 = []
            row2 = []
            continue
        line2 = line1[1].split("\n")
        row1.append(line1[0])
        row2.append(line2[0])
    return dataset

traindata = getdataset('data/train.txt')
testdata = getdataset('data/trial.txt')

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

word_to_ix = {}
for sent, tags in traindata:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
for sent, tags in testdata:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
tag_to_ix = {"B": 0, "I": 1, "O": 2}

# 实际中通常使用更大的维度如32维, 64维.
# 这里我们使用小的维度, 为了方便查看训练过程中权重的变化.
EMBEDDING_DIM = 300
HIDDEN_DIM = 150

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM以word_embeddings作为输入, 输出维度为 hidden_dim 的隐藏状态值
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True).to(device)

        # 线性层将隐藏状态空间映射到标注空间
        self.hidden2tag = nn.Linear(2 * hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 一开始并没有隐藏状态所以我们要先初始化一个
        # 关于维度为什么这么设计请参考Pytoch相关文档
        # 各个维度的含义是 (num_layers*num_directions, batch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim).to(device)),
                autograd.Variable(torch.zeros(2, 1, self.hidden_dim).to(device)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

# 设备使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix)).to(device)
model.load_state_dict(torch.load('best.pth'))

# 查看训练后的得分
with torch.no_grad():
    total = 0
    total_a = 0
    correct = 0
    cnt = 0
    correct_a = 0
    print('10 test in all traindata')
    for sentence, tags in testdata:
        inputs = prepare_sequence(sentence, word_to_ix).to(device)
        tag_scores = model(inputs).cpu()
        targets = prepare_sequence(tags, tag_to_ix)
        tag_scores = tag_scores.argmax(axis=1)
        correct += targets.eq(tag_scores).sum()
        if(targets.eq(tag_scores).sum().item() == len(targets)):
            correct_a += 1
        total += len(targets)
        total_a += 1

        if cnt < 20:
            cnt += 1
            item1 = " ".join(sentence)
            item2 = " ".join(tags)
            print('sentence is :', item1)
            print("it's label is :", item2)
            print('the predicted result is :')
            for i in tag_scores:
                print(classes[i.item()], " ", end="")
            print("\n")

    
    print('Acc(all aspect): ', torch.true_divide(correct, total).item())
    print('Acc(all sentence): ', correct_a / total_a)
