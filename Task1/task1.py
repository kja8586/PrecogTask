import numpy
import pandas
import cv2
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
# DataLoader wrapper over datasets and for batch size, Subset for creating subset of large dataset, random_split for splitting into different splits
from torchvision import transforms, datasets

from sklearn.metrics import precision_score, recall_score, f1_score

# Comman Functions
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train_val(model, train_loader, test_loader, device, opt, mode='train', epochs=5):
  criterion = nn.CrossEntropyLoss()
  model.to(device)

  if mode == 'train':
    for epoch in range(epochs):
      model.train()
      running_loss, epoch_loss = 0.0, 0.0; running_count = 0
      all_preds, all_labels = [], []

      for i, (x, y) in enumerate(train_loader):
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        opt.step()

        epoch_loss += loss.item() * x.size(0)
        running_loss += loss.item() * x.size(0)
        running_count += x.size(0)
        all_labels.append(y.cpu())
        all_preds.append(outputs.argmax(dim=1).cpu())

        if i % 10 == 0 :
          print(f"Epoch: {epoch+1}/{epochs} | Batch: {i}/{len(train_loader)} | BatchLoss: {running_loss/running_count:.5f}")
          running_loss, running_count = 0.0, 0.0

      all_labels = torch.cat(all_labels).numpy()
      all_preds = torch.cat(all_preds).numpy()
      epoch_loss = epoch_loss/len(train_loader.dataset)
      epoch_acc = 100*(all_preds == all_labels).sum()/len(all_labels)
      epoch_precision = precision_score(all_labels, all_preds, average='macro')
      epoch_recall = recall_score(all_labels, all_preds, average='macro')
      epoch_f1 = f1_score(all_labels, all_preds, average='macro')

      print(f"Epoch: {epoch+1}/{epochs} | Loss: {epoch_loss:.5f}, Accuracy: {epoch_acc:.2f}%, "
            f"Precision: {epoch_precision:.2f}, Recall: {epoch_recall:.2f}, F1 score: {epoch_f1:.2f}")
  else:
    model.eval()
    all_preds, all_labels = [], []
    test_loss = 0.0
    with torch.no_grad():
      for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        test_loss += criterion(output, y)*x.size(0)
        all_labels.append(y.cpu())
        all_preds.append(output.argmax(dim=1).cpu())

      all_labels = torch.cat(all_labels).numpy()
      all_preds = torch.cat(all_preds).numpy()
      test_loss /= len(test_loader.dataset)
      test_acc = 100*(all_preds == all_labels).sum()/len(all_labels)
      test_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
      test_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
      test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

      print(f"Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%"
            f"Test Precison: {test_precision:.2f}, Test Recall: {test_recall}, Test F1: {test_f1}")


class customDataset(Dataset):
  def __init__(self, data_dir, req_classes, req_img_per_class, transform=None):
    self.data_dir = data_dir
    self.transform = transform
    self.req_classes = req_classes
    self.req_img_per_class = req_img_per_class
    self.num_img_per_class = 100

    self.image_paths = []
    self.labels = []

    class_folders = sorted(os.listdir(self.data_dir))
    random.seed(42)
    selected_class = random.sample(class_folders, self.req_classes)

    for cls in selected_class:
      imgs = sorted(os.listdir(os.path.join(self.data_dir, cls)))
      imgs = imgs[:self.req_img_per_class]

      for img in imgs:
        self.image_paths.append(os.path.join(self.data_dir, cls, img))
        self.labels.append(cls)

    self.classes = selected_class
    self.num_classes = len(self.classes)
    unique_labels = sorted(set(self.labels))
    self.lbl2idx = {lbl:idx for idx, lbl in enumerate(unique_labels)}
    self.idx2lbl = {idx:lbl for lbl, idx in self.lbl2idx.items()}
    self.labels = [self.lbl2idx[lbl] for lbl in self.labels]
    self.samples = list(zip(self.image_paths, self.labels))

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    img = Image.open(self.image_paths[idx])
    label = self.labels[idx]

    if self.transform:
      img = self.transform(img)

    return img, label
  
# Easy set calssification
easy_dataset_path = "/mnt/sdb/users/kja8586/precog/Datasets/easySet" # Folder path of images

data_transforms = transforms.Compose([
    transforms.Resize((32,64)),
    transforms.GaussianBlur(3),
    transforms.ToTensor()
]) # This resizes the images, apply gaussian blur and convert to torch tensors

# 100 images class
total_data_100 = customDataset(easy_dataset_path, req_classes=100, req_img_per_class=100, transform=data_transforms)
train_size = int(0.8 * len(total_data_100)) # Splitting the size for train and test set
test_size = len(total_data_100) - train_size
# Splitting into train and test
train_dataset_100, test_dataset_100 = random_split(total_data_100, [train_size, test_size], generator=torch.Generator().manual_seed(42))
# Wrapping using DataLoader
train_loader_100 = DataLoader(train_dataset_100, batch_size=64, shuffle=True, num_workers=4)
test_loader_100 = DataLoader(test_dataset_100, batch_size=64, num_workers=4)

# 50 images per class
total_data_50 = customDataset(easy_dataset_path, req_classes=100, req_img_per_class=50, transform=data_transforms)
train_size = int(0.8 * len(total_data_50)) # Splitting the size for train and test set
test_size = len(total_data_50) - train_size
# Splitting into train and test
train_dataset_50, test_dataset_50 = random_split(total_data_50, [train_size, test_size], generator=torch.Generator().manual_seed(42))
# Wrapping using DataLoader
train_loader_50 = DataLoader(train_dataset_50, batch_size=64, shuffle=True, num_workers=4)
test_loader_50 = DataLoader(test_dataset_50, batch_size=64, num_workers=4)

# 30 images per class
total_data_30 = customDataset(easy_dataset_path, req_classes=100, req_img_per_class=30, transform=data_transforms)
train_size = int(0.8 * len(total_data_30)) # Splitting the size for train and test set
test_size = len(total_data_30) - train_size
# Splitting into train and test
train_dataset_30, test_dataset_30 = random_split(total_data_30, [train_size, test_size], generator=torch.Generator().manual_seed(42))# Wrapping using DataLoader
train_loader_30 = DataLoader(train_dataset_30, batch_size=64, shuffle=True, num_workers=4)
test_loader_30 = DataLoader(test_dataset_30, batch_size=64, num_workers=4)

# 10 images per class
total_data_10 = customDataset(easy_dataset_path, req_classes=100, req_img_per_class=10, transform=data_transforms)
train_size = int(0.8 * len(total_data_10)) # Splitting the size for train and test set
test_size = len(total_data_10) - train_size
# Splitting into train and test
train_dataset_10, test_dataset_10 = random_split(total_data_10, [train_size, test_size], generator=torch.Generator().manual_seed(42))# Wrapping using DataLoader
train_loader_10 = DataLoader(train_dataset_10, batch_size=64, shuffle=True, num_workers=4)
test_loader_10 = DataLoader(test_dataset_10, batch_size=64, num_workers=4)

# Model Definition
class EasyModel(nn.Module):
  def __init__(self):
    super(EasyModel, self).__init__()
    # Block 1
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    # Block 2
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    # FC block
    self.fc1 = nn.Linear(64*8*16, 512)
    self.fc2 = nn.Linear(512, 100)

  def forward(self, x):
    # Block 1
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.bn1(x)
    x = F.max_pool2d(x, 2)
    # Block 2
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.bn2(x)
    x = F.max_pool2d(x, 2)
    # FC
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    output = self.fc2(x)

    return output
  
# Train and eval 100 per class
model_100 = EasyModel()
opt = optim.Adam(params=model_100.parameters(), lr=1e-3)
train_val(model_100, train_loader_100, test_loader_100, device, opt ,epochs=50, mode='train')
train_val(model_100, train_loader_100, test_loader_100, device, opt, mode='else')
print(100)
# Train and eval 50 per class
model_50 = EasyModel()
opt = optim.Adam(params=model_50.parameters(), lr=1e-3)
train_val(model_50, train_loader_50, test_loader_50, device, opt, epochs=50, mode='train')
train_val(model_50, train_loader_50, test_loader_50, device, opt, mode='else')
print(50)
# Train and eval 30 per class
model_30 = EasyModel()
opt = optim.Adam(params=model_30.parameters(), lr=1e-3)
train_val(model_30, train_loader_30, test_loader_30, device, opt, epochs=50, mode='train')
train_val(model_30, train_loader_30, test_loader_30, device, opt, mode='else')
print(30)
# Train and eval 10 per class
model_10 = EasyModel()
opt = optim.Adam(params=model_10.parameters(), lr=1e-3)
train_val(model_10, train_loader_10, test_loader_10, device, opt, epochs=50, mode='train')
train_val(model_10, train_loader_10, test_loader_10, device, opt, mode='else')
print(10)

# Hard set classification
hard_dataset_path = "/mnt/sdb/users/kja8586/precog/Datasets/hardSet"

hard_data_transforms = transforms.Compose([
    transforms.Resize((32, 64)),
    #transforms.GaussianBlur(3),
    transforms.Grayscale(1),
    transforms.ToTensor()
])

# 100 images per class
total_hard_data_100 = customDataset(data_dir = hard_dataset_path, req_classes=100, req_img_per_class=100, transform=hard_data_transforms)
train_size = int(0.8 * len(total_hard_data_100))
test_size = len(total_hard_data_100) - train_size
hardTraindata_100, hardTestdata_100 = random_split(total_hard_data_100, [train_size, test_size], generator=torch.Generator().manual_seed(42))
hardTrainLoader_100 = DataLoader(hardTraindata_100, batch_size=64, shuffle=True, num_workers=4)
hardTestLoader_100 = DataLoader(hardTestdata_100, batch_size=64, num_workers=4)

# 50 images per class
total_hard_data_50 = customDataset(data_dir = hard_dataset_path, req_classes=100, req_img_per_class=50, transform=hard_data_transforms)
train_size = int(0.8 * len(total_hard_data_50))
test_size = len(total_hard_data_50) - train_size
hardTraindata_50, hardTestdata_50 = random_split(total_hard_data_50, [train_size, test_size], generator=torch.Generator().manual_seed(42))
hardTrainLoader_50 = DataLoader(hardTraindata_50, batch_size=64, shuffle=True, num_workers=4)
hardTestLoader_50 = DataLoader(hardTestdata_50, batch_size=64, num_workers=4)

# 30 images per class
total_hard_data_30 = customDataset(data_dir = hard_dataset_path, req_classes=100, req_img_per_class=30, transform=hard_data_transforms)
train_size = int(0.8 * len(total_hard_data_30))
test_size = len(total_hard_data_30) - train_size
hardTraindata_30, hardTestdata_30 = random_split(total_hard_data_30, [train_size, test_size], generator=torch.Generator().manual_seed(42))
hardTrainLoader_30 = DataLoader(hardTraindata_30, batch_size=64, shuffle=True, num_workers=4)
hardTestLoader_30 = DataLoader(hardTestdata_30, batch_size=64, num_workers=4)

# 10 images per class
total_hard_data_10 = customDataset(data_dir = hard_dataset_path, req_classes=100, req_img_per_class=10, transform=hard_data_transforms)
train_size = int(0.8 * len(total_hard_data_10))
test_size = len(total_hard_data_10) - train_size
hardTraindata_10, hardTestdata_10 = random_split(total_hard_data_10, [train_size, test_size], generator=torch.Generator().manual_seed(42))
hardTrainLoader_10 = DataLoader(hardTraindata_10, batch_size=64, shuffle=True, num_workers=4)
hardTestLoader_10 = DataLoader(hardTestdata_10, batch_size=64, num_workers=4)

# Hard model
class HardModel(nn.Module):
  def __init__(self):
    super(HardModel, self).__init__()
    # Block 1
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(32)

    # Block 2
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(128)

    # Block 3
    self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(256)

    # FC Block
    self.fc1 = nn.Linear(256, 512)
    self.fc2 = nn.Linear(512, 100)

  def forward(self, x):
    # Block 1
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(self.bn1(x), 2)
    # Block 2
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = F.max_pool2d(self.bn2(x), 2)
    # Block 3
    x = F.relu(self.conv5(x))
    x = F.relu(self.conv6(x))
    # FC Block
    x = F.adaptive_avg_pool2d(self.bn3(x), 1)
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    output = self.fc2(x)

    return output
  
# Train and eval 100 per class
hardModel_100 = HardModel()
opt = optim.Adam(params=hardModel_100.parameters(), lr=0.001)
train_val(hardModel_100, hardTrainLoader_100, hardTestLoader_100, device, opt, epochs=50, mode='train')
train_val(hardModel_100, hardTrainLoader_100, hardTestLoader_100, device, opt, mode='else')
print(100)
# Train and eval 50 per class
hardModel_50 = HardModel()
opt = optim.Adam(params=hardModel_50.parameters(), lr=0.001)
train_val(hardModel_50, hardTrainLoader_50, hardTestLoader_50, device, opt, epochs=50, mode='train')
train_val(hardModel_50, hardTrainLoader_50, hardTestLoader_50, device, opt, mode='else')
print(50)
# Train and eval 30 per class
hardModel_30 = HardModel()
opt = optim.Adam(params=hardModel_30.parameters(), lr=0.001)
train_val(hardModel_30, hardTrainLoader_30, hardTestLoader_30, device, opt, epochs=50, mode='train')
train_val(hardModel_30, hardTrainLoader_30, hardTestLoader_30, device, opt, mode='else')
print(30)
# Train and eval 10 per class
hardModel_10 = HardModel()
opt = optim.Adam(params=hardModel_10.parameters(), lr=0.001)
train_val(hardModel_10, hardTrainLoader_10, hardTestLoader_10, device, opt, epochs=50, mode='train')
train_val(hardModel_10, hardTrainLoader_10, hardTestLoader_10, device, opt, mode='else')
print(10)
