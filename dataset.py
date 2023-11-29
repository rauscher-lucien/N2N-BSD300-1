import os
import cv2
import torch

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.file_list = [filename for filename in os.listdir(data_dir) if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        self.transform = transform

    def __getitem__(self, index):
        filename = self.file_list[index]
        image = cv2.imread(os.path.join(self.data_dir, filename), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.file_list)

    
    