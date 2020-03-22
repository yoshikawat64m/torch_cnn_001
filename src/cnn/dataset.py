from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class MyDataset(Dataset):

    def __init__(self, label_file, image_dir, size=224):
        label_df = pd.read_csv(label_file)
        self.image_dir = image_dir
        self.image_files = label_df.iloc[:, 0]
        self.labels = label_df.iloc[:, 1]
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.image_files[idx]))
        label = int(self.labels[idx])

        image = self.transform(image.convert('RGB'))

        return image, label
