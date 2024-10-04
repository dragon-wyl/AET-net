import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, classification_report
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"device = {device}")


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor, min_val, max_val


def min_max_denormalize(normalized_tensor, min_val, max_val):
    denormalized_tensor = normalized_tensor * (max_val - min_val) + min_val
    return denormalized_tensor


def zscore_normalize(tensor):
    mean_val = tensor.mean()
    std_val = tensor.std()
    normalized_tensor = (tensor - mean_val) / std_val
    return normalized_tensor, mean_val, std_val


def zscore_denormalize(normalized_tensor, mean_val, std_val):
    denormalized_tensor = normalized_tensor * std_val + mean_val
    return denormalized_tensor


def data_prehandle(file=None, df=None, start=0, label="label", batch_size=32):

    if df is None:
        data = pd.read_csv(file)
    else:
        data = df
    data = data.iloc[:, start:]
    l = data.pop(label)
    data_tensor = torch.from_numpy(data.values).to(torch.float32)
    print(f"data.shape:{data.shape}, data_tensor:{data_tensor.shape}")
    nor_data_tensor, ae_mean, ae_std = zscore_normalize(data_tensor)
    dataset = TensorDataset(nor_data_tensor, nor_data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, data_tensor, nor_data_tensor, l, ae_mean, ae_std

def data_prehandle_min_max(file=None, df=None, start=0, label="label", batch_size=32):

    if df is None:
        data = pd.read_csv(file)
    else:
        data = df
    data = data.iloc[:, start:]
    l = data.pop(label)
    data_tensor = torch.from_numpy(data.values).to(torch.float32)
    print(f"data.shape:{data.shape}, data_tensor:{data_tensor.shape}")
    nor_data_tensor, ae_min, ae_max = min_max_normalize(data_tensor)
    dataset = TensorDataset(nor_data_tensor, nor_data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, data_tensor, nor_data_tensor, l, ae_min, ae_max

# AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Tanh(),
            nn.Linear(input_dim * 2, encoding_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim * 2),
            nn.Tanh(),
            nn.Linear(input_dim * 2, input_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_ae(model, dataloader, num_epochs=100, learning_rate=0.0001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward(retain_graph=True)
            optimizer.step()

        if epoch % 10 == 0:
            print(
                "Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item())
            )


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dim_feedforward=2048, nhead=8, nlayers=6, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.input_dim = input_dim
        self.src_mask = None
        self.encoder_num = 1024
        self.fc = nn.Linear(self.input_dim, self.encoder_num)
        self.softmax = nn.Softmax(dim=1)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_num,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.classifier = nn.Linear(self.encoder_num, num_classes)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
            self.src_mask = mask
        src = self.fc(src)
        src = src * math.sqrt(self.encoder_num)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.classifier(output)
        return torch.squeeze(output)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def train_transformer(num_epochs=500, para_file='best_model.pth', model=None, train_loader=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = num_epochs
    min_loss = 10000
    best_file = para_file
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_test_loss = 0
        num = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            num += 1
            data = data.unsqueeze(1).to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        epoch_loss = epoch_loss / num
        epoch_losses.append(epoch_loss)
        if (epoch_loss < min_loss):
            min_loss = epoch_loss
            print(f'save best model, epoch {epoch} loss = {min_loss}')
            torch.save(model.state_dict(), best_file)

        # print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss}')
        ### Testing
        model.eval()
        num = 0
        with torch.inference_mode():
            for batch_idx, (data_test, target_test) in enumerate(train_loader):
                num += 1
                # 1. Forward pass
                data_test = data_test.to(device)
                target_test = target_test.to(device)
                test_logits = model(data_test)
                test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
                # 2. Calculate test loss and accuracy
                test_loss = criterion(test_logits, target_test.to(device))
                epoch_test_loss += test_loss.item()
        epoch_test_loss = epoch_test_loss / num
        epoch_test_losses.append(epoch_test_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss}, Test Loss: {epoch_test_loss}')


def test_transformer(para_file="best_model.pth", model=None, test_loader=None):
    model.load_state_dict(torch.load(para_file))
    model.eval()
    total = 0
    correct = 0
    predicted_probs = []
    targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.unsqueeze(1).to(device)
            target = target.to(device)
            output = model(data)
            predicted_probs.extend(
                torch.softmax(output, dim=1).tolist()
            )  
            targets.extend(target.tolist())
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Accuracy of the model on the test set: {100 * correct / total}%")
    

    auc = roc_auc_score(targets, predicted_probs, multi_class="ovr")

    predicted_labels = np.argmax(predicted_probs, axis=1)
    recall = recall_score(targets, predicted_labels, average="macro")
    precision = precision_score(targets, predicted_labels, average="macro")

    f1 = f1_score(targets, predicted_labels, average="macro")

    print(f"AUC: {auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1 score: {f1}")

    return 100 * correct / total, auc, recall, precision, f1

def test_transformer_subclass(para_file="best_model.pth", model=None, test_loader=None):
    model.load_state_dict(torch.load(para_file))
    model.eval()
    total = 0
    correct = 0
    predicted_probs = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.unsqueeze(1).to(device)
            target = target.to(device)
            output = model(data)
            predicted_probs.extend(
                torch.softmax(output, dim=1).tolist()
            )  
            targets.extend(target.tolist())
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"Overall Accuracy of the model on the test set: {accuracy}%")

    auc = roc_auc_score(targets, predicted_probs, multi_class="ovr")
    print(f"Overall AUC: {auc}")

    predicted_labels = np.argmax(predicted_probs, axis=1)

    # Generate classification report for each class
    report = classification_report(targets, predicted_labels, output_dict=True)

    num_classes = len(set(targets))
    targets = np.array(targets)
    label_size = [0 for i in range(num_classes)]
    correct_label_size = [0 for i in range(num_classes)]
    
    for target, pred in zip(targets, predicted_labels):
        label_size[target] += 1
        if target == pred:
            correct_label_size[target] += 1
    print(f'{label_size=}')
    print(f'{correct_label_size=}')
    
    for class_label in range(num_classes):
        # Calculate per-class AUC
        
        class_probs = np.array(predicted_probs)[:, class_label]  # Probabilities for the current class
        class_auc = roc_auc_score((targets == class_label).astype(int), class_probs)
        class_label = str(class_label)
        print(f"*************************")
        print(f"  Class {class_label}:")
        print(f"  Precision: {report[class_label]['precision']:.4f}")
        print(f"  Recall: {report[class_label]['recall']:.4f}")
        print(f"  F1 Score: {report[class_label]['f1-score']:.4f}")
        print(f"  AUC: {class_auc:.4f}")
        print(f"*************************")
        print()
    return accuracy, auc

def train_ae(model, dataloader, num_epochs=100, learning_rate=0.0001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward(retain_graph=True)
            optimizer.step()

        if epoch % 10 == 0:
            print(
                "Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item())
            )


def count_num(y_train_tensor):
    unique_values = torch.unique(y_train_tensor)

    value_counts = torch.histc(
        y_train_tensor.float(),
        bins=len(unique_values),
        min=0,
        max=len(unique_values) - 1,
    )

    for i, value in enumerate(unique_values):
        count = int(value_counts[i])
        print(f"v: {value} ,num: {count}")
