import torch
import torchvision
from torchvision import transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='data', train=True, transform=None):
        super().__init__()
    
        data_dir = 'data'

        train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())

        size = len(train_dataset)

        train_data, val_data = super().random_split(train_dataset, [int(size - size * 0.2), int(size * 0.2)])

        batch_size = 256

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
        valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
    
    def __len__(self):
        return len(self.train_dataset) + len(self.valid_dataset) + len(self.test_dataset)
    
    def getDataLoader(self):
        return self.train_loader, self.valid_loader, self.test_loader
    
    