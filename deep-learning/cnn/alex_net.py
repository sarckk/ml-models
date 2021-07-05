"""
    PyTorch implementation of AlexNet (Krizhevsky et al., 2012)
"""

import torch
import requests
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class Conv2dWithReLU(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return F.relu(super().forward(x))

class Debug(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        print(x.shape)
        return x


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.00002
        
        # padding size for conv2d layers to make output size equal to input size is found by solving for (n + 2*pad - ks)//stride + 1 == desired_output_size
        self.layers = nn.Sequential(                    # Tensor shape afterwards (assuming 227 x 227 x 3 images)
                Conv2dWithReLU(3, 96, 11, 4),           # 96  x 55 x 55  
                nn.LocalResponseNorm(2, self.alpha),    # ''
                nn.MaxPool2d(3, 2),                     # 96  x 27 x 27 
                Conv2dWithReLU(96, 256, 5, padding=2),  # 256 x 27 x 27  --> padding = "SAME"
                nn.LocalResponseNorm(2, self.alpha),    # ''
                nn.MaxPool2d(3, 2),                     # 256 x 13 x 13 
                Conv2dWithReLU(256, 384, 3, padding=1), # 384 x 13 x 13  --> padding = "SAME"
                Conv2dWithReLU(384, 384, 3, padding=1), # 384 x 13 x 13  --> padding = "SAME"
                Conv2dWithReLU(384, 256, 3, padding=1), # 256 x 13 x 13  --> padding = "SAME"
                nn.MaxPool2d(3,2),                      # 256 x 6  x 6 
                nn.Flatten(),                           # 9216
                nn.Linear(256*6*6, 4096),               # 4096 
                nn.Dropout(0.5),                        # ''
                nn.ReLU(),                              # ''
                nn.Linear(4096, 4096),                  # 4096 
                nn.Dropout(0.5),                        # ''
                nn.ReLU(),                              # ''
                nn.Linear(4096, 101),                   # 101
        )

    def forward(self, x):
        return self.layers(x)


"""
    Functions and classes related to data processing and model training
"""

# Not used currently
def get_images_and_labels(root):
    top_level_dir = os.path.join(root, "101_ObjectCategories")
    categories = sorted(os.listdir(top_level_dir))
    categories.remove("BACKGROUND_Google")

    images = []
    labels = []

    for category_index, c in enumerate(categories):
        im_dir = os.path.join(top_level_dir, c)
        print(im_dir)
        nb_images = len(os.listdir(im_dir))
        print(nb_images)
        for image_index in range(nb_images):
            img = Image.open(os.path.join(im_dir, "image_{:04d}.jpg".format(image_index+1)))
            img = img.convert("RGB")
            images.append(img)
            labels.append(category_index)

    return np.array(images), np.array(labels), categories


class LazyLoadDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def sum_correct(pred, target):
    return (torch.argmax(pred, dim=1) == target).float().sum()


def train_minibatch(model, dl, loss_fn, opt, device):
    model.train()
    size = len(dl.dataset)
    train_loss = 0
    n_correct = 0

    for batch, (Xb, yb) in enumerate(dl):
        Xb, yb = Xb.to(device), yb.to(device)
        out = model(Xb)
        loss = loss_fn(out, yb)
       
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_loss += loss.item() * Xb.shape[0]
        n_correct += sum_correct(out, yb).item()
        
    train_loss /= size
    train_accuracy = n_correct / size
    
    print(f"loss: {train_loss:>0.3f} \t accuracy: {train_accuracy:>0.3f}")


def validate_minibatch(model, dl, loss_fn, device):
    model.eval()
    size = len(dl.dataset)
    n_correct = 0
    valid_loss = 0

    with torch.no_grad():
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            valid_loss += loss_fn(out, yb).item()
            n_correct += sum_correct(out, yb).item()

    valid_loss /= size
    valid_accuracy = n_correct / size

    print(f"valid_loss: {valid_loss:>0.3f} \t valid_accuracy: {valid_accuracy:>0.3f}")



def predict(url, model, categories, transform, device):
    new_image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    new_image = transform(new_image).to(device)
    new_image = new_image.reshape(1, 3, 227, 227)

    out = model(new_image)

    proba, category_index = out.topk(1, dim = 1)
    label = categories[category_index.item()]
    return proba.item(), label


if __name__ == "__main__":
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch import optim
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import StratifiedShuffleSplit

    learning_rate = 0.01
    n_epochs = 100
    BATCH_SIZE = 64
    normalize_stats = {'mean':[0.485, 0.456, 0.406],'std':[0.229, 0.224, 0.225]} 

    data_transforms = {
        'train': transforms.Compose([
                     transforms.Lambda(lambda x : x.convert('RGB')),
                     transforms.RandomResizedCrop(227),
                     transforms.RandomRotation(degrees=15),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(**normalize_stats)
                ]), 
        'valid': transforms.Compose([
                     transforms.Lambda(lambda x : x.convert('RGB')),
                     transforms.Resize((227,227)),
                     transforms.ToTensor(),
                     transforms.Normalize(**normalize_stats)
                ]),
    }
    
    # Lazy loading data
    # Refer to https://discuss.pytorch.org/t/using-imagefolder-random-split-with-multiple-transforms/79899/3 
    root = os.path.join('..', 'data')
    dataset    = datasets.Caltech101(root, download=True)

    train_ds   = LazyLoadDataset(dataset, transform=data_transforms['train'])
    val_ds     = LazyLoadDataset(dataset, transform=data_transforms['valid'])

    # Stratified sampling 
    indices    = np.arange(len(dataset))
    labels     = np.array(dataset.y)

    train_idx, rest     = next(StratifiedShuffleSplit(n_splits=1, test_size = 0.3, random_state=42).split(indices, labels))
    valid_idx, test_idx = next(StratifiedShuffleSplit(n_splits=1, test_size = 0.5, random_state=42).split(rest, labels[rest]))

    train_data = Subset(train_ds, indices=train_idx)
    valid_data = Subset(val_ds,   indices=valid_idx)
    test_data  = Subset(val_ds,   indices=test_idx)

    train_dl   = DataLoader(train_data, BATCH_SIZE, shuffle=True)
    val_dl     = DataLoader(valid_data, BATCH_SIZE, shuffle=True)
    test_dl    = DataLoader(test_data,  BATCH_SIZE, shuffle=True)

    # training the network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Running on ", device)

    print("============ AlexNet details ============")
    model = AlexNet().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch} / {n_epochs}")
        train_minibatch(model, train_dl, loss_fn, opt, device)
        validate_minibatch(model, val_dl, loss_fn, device)

    # Test final accuracy
    model.eval()
    model.to(device)
    with torch.no_grad():
            accs = [(torch.argmax(model(Xb.to(device)), dim=1) == yb.to(device)).float().mean() 
                    for Xb, yb in test_dl]
            test_accuracy = torch.stack(accs).float().mean()
            print(f"Test accuracy: {test_accuracy:>0.3f}")

            
    # Save model
    MODELS_PATH = os.path.join('..', 'models')
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    SAVE_PATH = os.path.join(MODELS_PATH, '_alexnet_0.pth')
    torch.save(model.state_dict(), SAVE_PATH)

    # Make new prediction
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    
    new_image_url = 'https://anthropocenemagazine.org/wp-content/uploads/2020/04/Panda-2.jpg'
    proba, label = predict(new_image_url, model, dataset.categories, data_transforms['valid'], device)
    print(f"The prediction is {label} with a probability of {proba}") # should be panda


    ######################################################################################################

    """
        Alternative method to load all images and labels up front
    """
    # X, y, categories = get_images_and_labels(root)

    # X_train, X_rest, y_train, y_rest = train_test_split(X, y, 
    #                                                    train_size=0.7, random_state=42, stratify=y)
    # X_val, X_test, y_val,  y_test = train_test_split(X_rest, y_rest, train_size=0.5, random_state=42)

    # train_ds   = LazyTransformDataset(X_train, y_train, transform=data_transforms['train'])
    # val_ds     = LazyTransformDataset(X_val,   y_val,   transform=data_transforms['valid'])

    # class LazyTransformDataset(Dataset):
    #     def __init__(self, images, labels, transform=None):
    #         self.images = images
    #         self.labels = labels
    #         self.transform = transform

    #     def __len__(self):
    #         return len(self.images)

    #     def __getitem__(self, index):
    #         image = self.images[index]
    #         label = self.labels[index]

    #         if self.transform is not None:
    #             image = self.transform(image)

    #         return image, label
    # Final test accuracy
    # model.eval()
    # with torch.no_grad():
    #     # Test final accuracy
    #     test_accuracy = (np.argmax(model(X_test), axis=1) == y_test).float().mean()
    #     print(f"Test accuracy: {test_accuracy:>0.3f}")
