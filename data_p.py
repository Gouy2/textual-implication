import re
import json
import os
from xml.etree import ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader

vocab_path = "./vocab.json"
train_path = "./train.xml"
test_path = "./test.xml"


class TextDataset(Dataset):
    def __init__(self, xml_file, vocab_path, max_length=80):
        with open (vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.pairs = []  # 存储(t1, t2, label)和(t2, t1, revlabel)的列表
        self.max_length = max_length
        self._load_and_process_data(xml_file)
        

    def _load_and_process_data(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for pair in root.iter('pair'):
            t1 = pair.find('t1').text if pair.find('t1') is not None else ''
            t2 = pair.find('t2').text if pair.find('t2') is not None else ''
            label = pair.get('label')
            revlabel = pair.get('revlabel')

            label = 1 if label == 'Y' else 0
            revlabel = 1 if revlabel == 'Y' else 0
            # 将文本转换为索引序列
            t1_indices = self.text_to_indices(t1, self.vocab, self.max_length)
            t2_indices = self.text_to_indices(t2, self.vocab, self.max_length)

            self.pairs.append((t1_indices, t2_indices, label))
            self.pairs.append((t2_indices, t1_indices, revlabel))

    
    def text_to_indices(self, text, char_to_index, max_length):
        # 初始化序列列表，包含开始标识符索引
        indices = [char_to_index["<start>"]]
        
        # 将文本中的每个字符转换为索引
        for char in text:
            if char in char_to_index:
                indices.append(char_to_index[char])
            else:
                indices.append(char_to_index["<unk>"])  # 未知字符替换
        
        # 添加结束标识符索引
        indices.append(char_to_index["<end>"])
        
        # 如果指定了最大长度，则根据需要截断或填充序列
        if max_length:
            # 填充
            while len(indices) < max_length:
                indices.append(char_to_index["<pad>"])
            # 截断
            indices = indices[:max_length]
        
        return indices


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        t1_indices, t2_indices, label = self.pairs[idx]
        return torch.tensor(t1_indices, dtype=torch.long), torch.tensor(t2_indices, dtype=torch.long), label


# train_dataset = TextDataset(train_path, vocab_path)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# print(len(train_dataset))
# print(train_dataset[101])

