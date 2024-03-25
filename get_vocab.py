import re
import json
import os
from xml.etree import ElementTree as ET

vocab_path = "./vocab.json"
train_path = "./train.xml"
test_path = "./test.xml"

# 提取中文字符
def extract_chinese(text):
   
    pattern = r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]'
    return re.findall(pattern, text)

def get_xml(file_path):
    # 解析XML文件
    tree = ET.parse(file_path)
    # 获取根元素
    root = tree.getroot()
    all_text = ""
    for pair in root.iter('pair'): # 找到所有pair标签
        t1 = pair.find('t1').text if pair.find('t1') is not None else '' 
        t2 = pair.find('t2').text if pair.find('t2') is not None else ''
        all_text += t1 + t2  # 将t1和t2的文本拼接起来
    return all_text


all_text = get_xml(train_path)+get_xml(test_path)  # 获取所有文本

chinese_list = extract_chinese(all_text)  # 提取中文字符

numbers = [str(i) for i in range(10)]  # 数字字符

# 删除重复项
unique_characters = list(set(chinese_list))
unique_characters += numbers  # 添加数字字符

unique_characters.sort()

special_tokens = ["<pad>", "<unk>", "<start>", "<end>"]
vocab_combined = special_tokens + unique_characters

# 转换为 {word: index} 形式的词典
vocab = {word: index for index, word in enumerate(vocab_combined)}

# 保存词典
with open(vocab_path, 'w',encoding='utf-8') as fw:
    json.dump(vocab, fw, ensure_ascii=False, indent=4)

#print(unique_characters[:100], len(unique_characters))
    

