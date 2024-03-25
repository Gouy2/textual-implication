import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes=2, dropout_rate=0.2):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 4, num_classes)  # 乘4因为是双向LSTM

    def forward(self, premise, hypothesis):
        # 嵌入层
        premise_embed = self.embedding(premise)
        hypothesis_embed = self.embedding(hypothesis)

        # 应用dropout到嵌入层的输出上
        premise_embed = self.dropout(premise_embed)
        hypothesis_embed = self.dropout(hypothesis_embed)

        # LSTM层
        premise_output, _ = self.lstm(premise_embed)
        hypothesis_output, _ = self.lstm(hypothesis_embed)

        # print(premise_output.shape, hypothesis_output.shape)

        # 获取最后时间步的输出用于分类
        premise_last_output = premise_output[:, -1, :]
        hypothesis_last_output = hypothesis_output[:, -1, :]

        # print(premise_last_output.shape, hypothesis_last_output.shape)

        # 拼接两个句子的表示
        combined_output = torch.cat((premise_last_output, hypothesis_last_output), dim=1)

        # print(combined_output.shape)

        # 全连接层和softmax输出
        output = self.fc(combined_output)
        return output
