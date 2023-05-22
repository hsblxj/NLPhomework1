import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os
import torch
import jieba


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        """
        将单词添加到字典中，建立单词到索引的映射关系
        """
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word


class Corpus(object):
    def __init__(self, filepath):
        self.dictionary = Dictionary()
        self.file_list = self.get_file(filepath)

    def get_file(self, filepath):
        """
        获取指定目录下的所有txt文件路径列表
        """
        txtlist = os.listdir(filepath)
        pathlist = []
        for name in txtlist:
            if name.lower().endswith('.txt') and name != 'inf.txt':
                path = os.path.join(filepath, name)
                pathlist.append(path)
        return pathlist

    def process_line(self, line):
        """
        处理文本行，去除空格和制表符
        """
        line = line.replace(' ', '').replace('\u3000', '').replace('\t', '')
        return line

    def process_file(self, path):
        """
        读取文件并处理文件内容
        """
        with open(path, 'r', encoding="ANSI") as f:
            lines = f.readlines()
            lines = [self.process_line(line) for line in lines]
            return ' '.join(lines)

    def get_data(self, batch_size):
        """
        获取数据集，并构建字典
        """
        tokens = 0
        data = ''
        for path in self.file_list:
            print(path)
            file_data = self.process_file(path)
            data += file_data + ' '
            tokens += len(file_data.split())
            for word in file_data.split():
                self.dictionary.add_word(word)

        ids = torch.LongTensor(tokens)
        token = 0
        for word in data.split():
            ids[token] = self.dictionary.word2idx[word]
            token += 1

        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        ids = ids.view(batch_size, -1)
        return ids


class LSTMmodel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMmodel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # 单词总数，每个单词的特征个数
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # 单词特征数，隐藏节点数，隐藏层数
        self.linear = nn.Linear(hidden_size, vocab_size)  # 全连接层

    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.linear(out)
        return out, (h, c)


def generate_text(model, corpus, num_samples):
    """
    使用训练好的模型生成文本
    """
    generated_text = []

    state = (torch.zeros(num_layers, 1, hidden_size).to(device),
             torch.zeros(num_layers, 1, hidden_size).to(device))

    prob = torch.ones(vocab_size)

    _input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

    for i in range(num_samples):
        output, state = model(_input, state)
        prob = output.exp()
        word_id = torch.multinomial(prob, num_samples=1).item()
        _input.fill_(word_id)
        word = corpus.dictionary.idx2word[word_id]
        word = '\n' if word == '<eos>' else word
        generated_text.append(word)

    generated_text = jieba.lcut(''.join(generated_text))
    generated_text = generated_text[:num_samples]

    return generated_text


if __name__ == "__main__":
    batch_size = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus = Corpus(r".\jyxstxtqj_downcc.com")
    ids = corpus.get_data(batch_size)
    vocab_size = len(corpus.dictionary)

    embed_size = 256
    hidden_size = 1024
    num_layers = 3
    num_epochs = 10
    seq_length = 30
    learning_rate = 0.001

    model = LSTMmodel(vocab_size, embed_size, hidden_size, num_layers).to(device)
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))

        for i in tqdm(range(0, ids.size(1) - seq_length, seq_length)):
            inputs = ids[:, i:i + seq_length].to(device)
            targets = ids[:, (i + 1):(i + 1) + seq_length].to(device)

            states = [state.detach() for state in states]

            outputs, states = model(inputs, states)
            loss = cost(outputs, targets.reshape(-1))

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    '''Save'''
    save_dir = './model_path'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model.pt')
    torch.save(model, save_path)

    '''Generation'''
    num_samples = 500
    generated_text = generate_text(model, corpus, num_samples)

    output_dir = './generation'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{num_samples}.txt')
    with open(output_file, 'w', encoding='utf-8') as gen_file:
        gen_file.write(''.join(generated_text))
