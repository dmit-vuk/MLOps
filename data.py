from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import ToTensor
from typing import Tuple

def get_dataloaders(batch_size: int) -> Tuple[DataLoader]:
    train_dataset = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    test_dataset = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    return train_dataloader, test_dataloader

class MNISTDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
