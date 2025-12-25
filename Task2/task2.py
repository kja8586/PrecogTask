import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms, datasets

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def ctc_greedy_decode(logits, blank=0):
    preds = logits.argmax(2)
    preds = preds.permute(1, 0)

    decoded = []
    for seq in preds:
        prev = blank
        out = []
        for p in seq:
            p = p.item()
            if p != blank and p != prev:
                out.append(p)
            prev = p
        decoded.append(out)
    return decoded

def edit_distance(s1, s2):
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[-1][-1]

def compute_cer(preds, targets):
    total_edits = 0
    total_chars = 0

    for p, t in zip(preds, targets):
        total_edits += edit_distance(p, t)
        total_chars += len(t)

    return total_edits / max(1, total_chars)

def evaluate_cer(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels, lengths in dataloader:
            imgs = imgs.to(device)

            logits = model(imgs)  # [T, B, C]
            preds = ctc_greedy_decode(logits)

            labels = labels.cpu()
            lengths = lengths.cpu()

            idx = 0
            targets = []
            for l in lengths:
                targets.append(labels[idx:idx+l].tolist())
                idx += l

            all_preds.extend(preds)
            all_targets.extend(targets)

    cer = compute_cer(all_preds, all_targets)
    return cer



def train_val(model, train_loader, test_loader, device, opt, mode='train', epochs=10):
  criterion = nn.CTCLoss(blank=0)
  model.to(device)
  if mode=='train':
    model.train()
    epoch_loss = 0.0
    for epoch in range(epochs):
      for i, (imgs, labels, lengths) in enumerate(train_loader):
        imgs, labels, lengths = imgs.to(device), labels.to(device), lengths.to(device)
        opt.zero_grad()
        logits = model(imgs)
        log_prob = logits.log_softmax(2)
        input_lengths = torch.LongTensor([log_prob.size(0)] * log_prob.size(1))
        loss = criterion(log_prob, labels, input_lengths, lengths)
        epoch_loss += loss.item()
        loss.backward()
        opt.step()
      epoch_loss /= len(train_loader)
      cer = evaluate_cer(model, test_loader, device)
      model.train()
      print(f"Epoch: {epoch+1}/{epochs} | Loss: {epoch_loss:.5f}| CER: {cer:.4f}")

  else:
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
      for i, (imgs, labels, lengths) in enumerate(test_loader):
        imgs, labels, lengths = imgs.to(device), labels.to(device), lengths.to(device)
        logits = model(imgs)
        log_prob = logits.log_softmax(2)
        input_lengths = torch.LongTensor([log_prob.size(0)] * log_prob.size(1))
        loss = criterion(log_prob, labels, input_lengths, lengths)
        test_loss += loss.item()

      test_loss /= len(test_loader)
      print(f"Test Loss: {test_loss:.5f}")

class CustomDataset(Dataset):
  def __init__(self, data_dir, img_transform=None):
    self.data_dir = data_dir
    self.img_transform = img_transform

    self.image_path = []
    self.labels = []
    self.labels_idx = []

    self.chars = string.ascii_uppercase + string.ascii_lowercase
    self.blank_idx = 0
    self.char2idx = {c: i+1 for i, c in enumerate(self.chars)}  
    self.idx2char = {i+1: c for i, c in enumerate(self.chars)}

    for i in os.listdir(self.data_dir):
      for j in os.listdir(os.path.join(self.data_dir, i)):
        self.image_path.append(os.path.join(self.data_dir, i, j))
        self.label = re.match(r"([a-zA-Z]+)_", j).group(1)
        self.labels.append(self.label)
        self.labels_idx.append([self.char2idx[c] for c in self.label])
    self.num_classes = len(self.chars) + 1



  def __len__(self):
    return len(self.image_path)

  def __getitem__(self, idx):
    img = Image.open(self.image_path[idx])
    label = self.labels[idx]
    labels_idx = self.labels_idx[idx]

    if self.img_transform:
      img = self.img_transform(img)

    return img, labels_idx, len(labels_idx)

  def collate_fn(self, batch):
    imgs, labels, lengths = zip(*batch)

    imgs = torch.stack(imgs, dim=0)
    labels = [torch.tensor(label, dtype=torch.long) for label in labels]
    lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    labels = torch.cat(labels, dim=0)

    return imgs, labels, lengths

hard_data_dir = "/mnt/sdb/users/kja8586/precog/Datasets/hardSet"
hard_transforms = transforms.Compose([
    transforms.Resize((32, 64)),
    transforms.ToTensor()
])

Harddataset = CustomDataset(data_dir=hard_data_dir, img_transform=hard_transforms)

train_size = int(0.8 * len(Harddataset))
test_size = len(Harddataset) - train_size

hard_train_dataset, hard_test_dataset = random_split(Harddataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

hardTrainLoader = DataLoader(hard_train_dataset, batch_size=64, shuffle=True, collate_fn=Harddataset.collate_fn)
hardTestLoader = DataLoader(hard_test_dataset, batch_size=64, collate_fn=Harddataset.collate_fn)

class CRNN(nn.Module):
  def __init__(self, num_classes):
    super(CRNN, self).__init__()
    # Conv Backbone
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(128)
    self.flat = nn.Linear(128*8, 256)
    self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, bidirectional=True)
    self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, bidirectional=True)
    self.fc = nn.Linear(2*128, num_classes)

  def forward(self, x):
    x = self.bn1(F.relu(self.conv2(F.relu(self.conv1(x)))))
    x = F.max_pool2d(x, 2)
    x = self.bn2(F.relu(self.conv4(F.relu(self.conv3(x)))))
    x = F.max_pool2d(x, 2)
    x = x.view(x.size(0), 128*8, 16) # Batch_size, channels*hieght, width
    x = x.permute(2, 0, 1) # [width, batch_size, feeatures] and width will be sequence length for LSTM
    x = F.relu(self.flat(x))
    x, _ = self.lstm1(x) # x is required output, _ has tuple(h_n, c_n) final hidden state and final cell state
    x, _ = self.lstm2(x)
    output = self.fc(x)
    return output
model = CRNN(num_classes=Harddataset.num_classes)
opt = optim.Adam(model.parameters(), lr=0.001)

train_val(model, hardTrainLoader, hardTestLoader, device, opt, epochs=50)
torch.save(model.state_dict(), "crnn_final.pth")
train_val(model, hardTrainLoader, hardTestLoader, device, opt, mode='else')
