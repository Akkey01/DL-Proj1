import numpy as np
import os
import pickle
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class CustomCIFAR10Dataset(Dataset):
    def __init__(self, root, mode, pkl_file_path=None, transform=None):

        self.mode = mode
        self.transform = transform

        if self.mode=='test_nolabel':
            if pkl_file_path is not None:
                with open(os.path.join(root, pkl_file_path), 'rb') as fo:
                    data_dict = pickle.load(fo, encoding='bytes')

                self.images = data_dict[b'data']
                self.labels = None
                self.ids = np.array(data_dict[b'ids'])
            else:
                raise ValueError('Pickle file path is missing!')
            
        elif self.mode=='train':
            train_images = []
            train_labels = []
            for i in range(1, 6):
                with open(os.path.join(root, 'cifar-10-batches-py','data_batch_{}'.format(i)), 'rb') as fo:
                    train_batch = pickle.load(fo, encoding='bytes')
                train_images.append(train_batch[b'data'])
                train_labels += train_batch[b'labels']

            train_images = np.vstack(train_images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Convert to HWC format
            train_labels = np.array(train_labels)

            self.images = train_images
            self.labels = train_labels
            self.ids = None

        elif self.mode=='test': 
            with open(os.path.join(root, 'cifar-10-batches-py','test_batch'), 'rb') as fo:
                test_dict = pickle.load(fo, encoding='bytes')

            test_images = test_dict[b'data']
            test_images = test_images.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)  # Convert to HWC format
            test_labels = np.array(test_dict[b'labels'])

            self.images = test_images
            self.labels = test_labels
            self.ids = None
        else:
            raise ValueError('Mode does not exist!')
         
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
            
        if self.mode=='train' or self.mode=='test':
            label = self.labels[idx]
            return img, label
        
        elif self.mode=='test_nolabel':
            id = self.ids[idx]
            return img, id
    
    def return_labels(self):
        return self.labels
    
    def return_ids(self):
        return self.ids

        
transform_train_old = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616)),
    transforms.Normalize((0.4438,0.4361,0.4058),(0.2775,0.2736,0.2819)),
    transforms.Normalize((0.1767,0.1746,0.1671),(0.8248,0.8233,0.8016)),
    transforms.Normalize((0.0596, 0.0582, 0.0527),(1.0142, 1.0138, 1.0085)),

    ])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616)),
    transforms.Normalize((0.4438,0.4361,0.4058),(0.2775,0.2736,0.2819)),
    transforms.Normalize((0.1767,0.1746,0.1671),(0.8248,0.8233,0.8016)),
    transforms.Normalize((0.0596, 0.0582, 0.0527),(1.0142, 1.0138, 1.0085)),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
])