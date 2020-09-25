import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
import torch

transform = transforms.Compose([transforms.ToTensor(), # to [0, 1]
                                # transforms.Normalize(mean=.5, std=.5) # to [-1, 1]
                                ])

class ImgDataset(Dataset):
    def __init__(self, root='../dataset', mode='train'):
        self.root_path = os.path.join(root, mode) # path of train/test set
        self.classes = [os.path.join(self.root_path, i)
                        for i in os.listdir(self.root_path)] # path of real/manipulated set
        self.imgs = []
        for index in range(len(self.classes)):
            for j in os.listdir(self.classes[index]):
                self.imgs.append(os.path.join(self.classes[index], j)) # path of images
        count_samples = np.int(len(self.imgs) / 2)
        self.labels = np.zeros(count_samples, dtype=np.long).tolist() + np.ones(count_samples, dtype=np.long).tolist()
        self.transforms = transform
        # self.x = np.array(x, dtype=np.float32)
        # self.y = np.array(y, dtype=np.int64)
    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_label = self.labels[index]
        img_load = Image.open(img_path).convert('L').resize((256, 256)) # convert to gray image, resize to (256, 256)
        if self.transforms:
            img_data = self.transforms(img_load)
        else:
            img_temp = np.asarray(img_load)
            img_data = torch.from_numpy(img_temp)
        # return self.x[index], self.y[index]
        return img_data, img_label
    def __len__(self):
        # return len(self.x)
        return len(self.imgs)

class Data():
    def __init__(self, conf):
        self.conf = conf
        self.data_path = conf.data_path
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

    # def extract_data(self, kind="train"):
    #     # you should load image here
    #     images = np.random.randn(100, 1, 256, 256)
    #     labels = np.ones(shape=(100))
    #     return images, labels

    def load_data(self, batch_size=0):
        print("-> load data from: {}".format(self.data_path))
        if not batch_size:
            batch_size = self.conf.batch_size
        # x, y = self.extract_data(kind="train")
        self.train_loader = DataLoader(ImgDataset(root=self.data_path, mode="train"), batch_size=batch_size, shuffle=True)
        # x, y = self.extract_data(kind="test")
        self.test_loader = DataLoader(ImgDataset(root=self.data_path, mode="test"), batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(ImgDataset(root=self.data_path, mode='val'), batch_size=batch_size, shuffle=True)
        return self

if __name__ == '__main__':
    img_dataset = ImgDataset(root='../dataset', mode='train')