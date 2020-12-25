import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
import torch.autograd as autograd

classes = {'O':-1, 'B':0, 'I':1}
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
torch.save(model.state_dict(), 'best.pth')
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 查看训练前的分数
# 注意: 输出的 i,j 元素的值表示单词 i 的 j 标签的得分
# 这里我们不需要训练不需要求导所以使用torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(traindata[0][0], word_to_ix).to(device)
    tag_scores = model(inputs)
    predicted = torch.max(tag_scores,1)
    print(tag_scores)

for epoch in range(20):  # 实际情况下你不会训练300个周期, 此例中我们只是随便设了一个值
    sum_loss = 0
    for sentence, tags in traindata:
        # 第一步: 请记住Pytorch会累加梯度.
        # 我们需要在训练每个实例前清空梯度
        model.zero_grad()

        # 此外还需要清空 LSTM 的隐状态,
        # 将其从上个实例的历史中分离出来.
        model.hidden = model.init_hidden()

        # 准备网络输入, 将其变为词索引的 Tensor 类型数据
        sentence_in = prepare_sequence(sentence, word_to_ix).to(device)
        targets = prepare_sequence(tags, tag_to_ix).to(device)

        # 第三步: 前向传播.
        tag_scores = model(sentence_in)

        # 第四步: 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
        loss = loss_function(tag_scores, targets)
        sum_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('loss: ', sum_loss)

torch.save(model.state_dict(), 'best.pth')
# 查看训练后的得分
with torch.no_grad():
    total = 0
    correct = 0
    for sentence, tags in traindata[100:]:
        inputs = prepare_sequence(sentence, word_to_ix).to(device)
        tag_scores = model(inputs).cpu()
        targets = prepare_sequence(tags, tag_to_ix)
        tag_scores = tag_scores.argmax(axis=1)
        correct += targets.eq(tag_scores).sum()
        total += len(targets)
    
    print('Acc: ', torch.true_divide(correct, total).item())