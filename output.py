import json
import torch
from xml.etree import ElementTree as ET
from torch.utils.data import Dataset
from data_p import TextDataset
import os
from model import LSTM


def predict_entailment(premise, hypothesis, model_path, vocab_path, max_length=80):
    # 加载词汇表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # 预处理文本
    premise_indices = text_to_indices(premise, vocab, max_length)
    hypothesis_indices = text_to_indices(hypothesis, vocab, max_length)
    
    # 将处理过的数据转换为张量，并添加批次维度
    premise_tensor = torch.tensor([premise_indices], dtype=torch.long)
    hypothesis_tensor = torch.tensor([hypothesis_indices], dtype=torch.long)
    
    # 使用模型进行推理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM(1500, 512, 128, 2, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(premise_tensor, hypothesis_tensor)
        prediction = torch.argmax(outputs, dim=1)
        
    # 解析模型输出
    return "蕴含" if prediction.item() == 1 else "不蕴含"

def text_to_indices(text, char_to_index, max_length):
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

if __name__ == "__main__":

    vocab_path = "./vocab.json"
    model_path = "./model/best_model.pth"
    premise = "约瑟夫·傅立叶被广泛公认为温室效应的发现者。"
    hypothesis = "约瑟夫·傅立叶自认为温室效应的发现者。"
    result_1 = predict_entailment(premise, hypothesis, model_path, vocab_path)
    result_2 = predict_entailment(hypothesis, premise, model_path, vocab_path)
    print("正向预测:", result_1)
    print("反向预测:", result_2)
