from torchvision import datasets
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

def data_transform():
    train_transforms = A.Compose([
        A.Normalize(mean = (0.49139968, 0.48215841, 0.44653091), std = (0.24703223, 0.24348513, 0.26158784)),
        A.HorizontalFlip(),
        A.ShiftScaleRotate(),
        A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=0.47),
        A.ToGray()
    ])

    test_transforms = A.Compose([
        A.Normalize(mean = (0.49421428, 0.48513139, 0.45040909), std = (0.24665252, 0.24289226, 0.26159238))
    ])

    return train_transforms, test_transforms

class Cifar10Dataset(datasets.CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None, viz=False):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.viz = viz

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=np.array(image))
            image = transformed["image"]
            if self.viz:
              return image, label
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)
        return image, label
    
class MNISTDataset(datasets.MNIST):
    def __init__(self, root="./data", train=True, download=True, transform=None, viz=False):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.viz = viz

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=np.array(image))
            image = transformed["image"]
            if self.viz:
              return image, label
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)
        return image, label

def dataloader(data, train_batch_size=None, val_batch_size=None, seed=42):
    train_transforms, test_transforms = data_transform()
    if data == "MNIST":
        train_ds = MNISTDataset('./data', train=True, download=True, transform=train_transforms)
        test_ds = MNISTDataset('./data', train=False, download=True, transform=test_transforms)
    elif data == "CIFAR10":
        train_ds = Cifar10Dataset('./data', train=True, download=True, transform=train_transforms)
        test_ds = Cifar10Dataset('./data', train=False, download=True, transform=test_transforms)

    cuda = torch.cuda.is_available()

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    train_batch_size = train_batch_size or (128 if cuda else 64)
    val_batch_size = val_batch_size or (128 if cuda else 64)

    train_dataloader_args = dict(shuffle=True, batch_size=train_batch_size, num_workers=4, pin_memory=True)
    val_dataloader_args = dict(shuffle=True, batch_size=val_batch_size, num_workers=4, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(train_ds, **train_dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_ds, **val_dataloader_args)

    return train_loader, test_loader

def data_details(cols=5, rows=2, train_data=True, transform=False):
    train_transforms, test_transforms = data_transform()
    if transform and train_data:
        transform = train_transforms
    elif transform and not(train_data):
        transform = test_transforms
    else:
        transform = None
    cifar10_ex = Cifar10Dataset('./data', train=train_data, download=True, transform=transform, viz=True )
    figure = plt.figure(figsize=(cols*1.5, rows*1.5))
    for i in range(1, cols * rows + 1):
        img, label = cifar10_ex[i]

        figure.add_subplot(rows, cols, i)
        plt.title(cifar10_ex.classes[label])
        plt.axis("off")
        plt.imshow(img, cmap="gray")

    plt.tight_layout()
    plt.show()
    if transform is None:
        print(' - mean:', np.mean(cifar10_ex.data, axis=(0,1,2)) / 255.)
        print(' - std:', np.std(cifar10_ex.data, axis=(0,1,2)) / 255.)
        print(' - var:', np.var(cifar10_ex.data, axis=(0,1,2)) / 255.)
    return cifar10_ex.classes