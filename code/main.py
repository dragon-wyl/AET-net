import nnutil
import torch 
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import math
device = nnutil.device
device

gen_met = pd.read_csv("TCGA-BRCA-Integrated-Data-Clear.csv")
gen_met


print(gen_met.shape)
integration_dimensions = [2048, 1024, 512, 256]
dim_losses = []
input_dim = gen_met.shape[1]
dataloader, data_tensor, nor_data_tensor, l, ae_mean, ae_std = nnutil.data_prehandle(df=gen_met)

for dimension in integration_dimensions:
    print(f"handling dimension: {dimension}")
    ae_model = AutoEncoder(data_tensor.shape[1], dimension).to(device)
    print(ae_model)
    dim_loss = train_ae(ae_model, dataloader=dataloader, learning_rate=0.0001)
    dim_losses.append(dim_loss)