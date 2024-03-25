import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import LSTM
from data_p import TextDataset
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, random_split


vocab_path = "./vocab.json"
train_path = "./train.xml"
test_path = "./test.xml"

vocab_size = 1500  
embed_size = 512
hidden_size = 128
num_layers = 2
num_classes = 2  # 对于二分类任务
learning_rate = 0.001
batch_size = 32
num_epochs = 100


train_dataset = TextDataset(train_path, vocab_path)

total_size = len(train_dataset)

# 制造验证集
val_size = int(total_size * 0.1)
train_size = total_size - val_size

# 随机分割数据集
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# 创建对应的DataLoader
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)


# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = LSTM(vocab_size, embed_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 假设保存模型的目录
model_save_path = './model'
log_file_path = os.path.join(model_save_path, 'log.txt')  # 定义log文件的路径
os.makedirs(model_save_path, exist_ok=True)

best_val_accuracy = 0  # 初始化最佳验证集准确率

with open(log_file_path, 'a') as log_file:  # 以追加模式打开log文件

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (premise, hypothesis, labels) in enumerate(train_loader):
            premise, hypothesis, labels = premise.to(device), hypothesis.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(premise, hypothesis)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 添加准确率计算
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            
            if (batch_idx + 1) % 100 == 0:  
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = 100 * correct / total
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.2f}%')

        model.eval()  # 将模型设置为评估模式
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # 不计算梯度，以加速和节省内存
            for premise, hypothesis, labels in val_loader:
                premise, hypothesis, labels = premise.to(device), hypothesis.to(device), labels.to(device)
                outputs = model(premise, hypothesis)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        if (epoch + 1) % 1 == 0:
                log_msg = f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {avg_accuracy:.2f}%, ' \
                        f'Validation Accuracy: {val_accuracy:.2f}%\n'
                log_file.write(log_msg)  # 将日志信息写入文件
                # print(log_msg, end='')
                print(f'Validation - Accuracy: {val_accuracy:.2f}%')

        # 可选：保存性能最佳的模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_path = os.path.join(model_save_path, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            # print(f'Best model saved to {save_path}')


