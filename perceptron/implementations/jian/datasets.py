import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import torch

###### download dataset
path = kagglehub.dataset_download("gpreda/chinese-mnist")
print("Path to dataset files:", path)
df = pd.read_csv(os.path.join(path, "chinese_mnist.csv"))
print(df.info())
print(f"number of classes: {len(df['value'].unique())}")

###### split dataset
df["label"] = df["code"] - 1

# 1) Train + temp split
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,            # 70% train, 30% temp
    stratify=df["label"],
    random_state=42
)

# 2) Split temp into validation and test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,            # half of the remainder
    stratify=temp_df["label"],
    random_state=42
)
print(f"samples in training set: {len(train_df)}")
print(f"samples in validation set: {len(val_df)}")
print(f"samples in testing set: {len(test_df)}")

print("Class staistics in Training set: ")
print(train_df['label'].value_counts().sort_index())
print("Class staistics in Validation set: ")
print(val_df['label'].value_counts().sort_index())
print("Class staistics in Testing set: ")
print(test_df['label'].value_counts().sort_index())

####### create dataset
# pytorch dataset help class and transform functions
# this class construction codes are obtained from https://www.kaggle.com/code/tomasrando/perceptron-multicapa-con-dropout
# thank Tomas Rando for sharing his work

class ChineseNumbersDataset(Dataset):
    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_name = f"input_{row['suite_id']}_{row['sample_id']}_{row['code']}.jpg"
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row['code']) - 1

        return image, label

# since three channels are all the same...only use one channel
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),        # a tensor
    # transforms.Normalize((0.5,), (0.5,)) # normalizar
])
img_dir = os.path.join(path, "data/data/")
train_dataset = ChineseNumbersDataset(train_df, img_dir, transform=transform)
val_dataset   = ChineseNumbersDataset(val_df,   img_dir, transform=transform)
test_dataset  = ChineseNumbersDataset(test_df,  img_dir, transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# a helper function to randomly plot an image from any dataset created
def img_plot(dataset, idx=None):
    if idx is None:
        idx = random.randint(0, len(train_dataset) - 1)

    img, label = dataset[idx]
    img_np = img.squeeze(0).numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(img_np, cmap='gray')  # clip in case of slight float errors
    plt.axis("off")
    plt.title(f"Label (zero-based): {label}\nOriginal code: {label + 1}")
    plt.show()
