from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class MyDataset(Dataset):

    def __init__(self, csv_path, root_dir, size=224):
        dataset_df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.image_files = dataset_df.iloc[:, 0]
        self.labels = dataset_df.iloc[:, 1]
        self.transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.image_files[idx]))
        label = int(self.labels[idx])

        image = self.transform(image.convert('RGB'))
        
        return image, label