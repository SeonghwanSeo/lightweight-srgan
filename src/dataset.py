import os
from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import RandomCrop, ToTensor, CenterCrop, Resize, InterpolationMode, RandomHorizontalFlip, RandomVerticalFlip, Compose
from torchvision.transforms.functional import rotate, resize
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def to_tensor(image) :
    tensor = ToTensor()(image)
    if tensor.size(0) == 1 :
        tensor = tensor.repeat(3,1,1)
    return tensor

class TrainDataset(Dataset):
    def __init__(
            self,
            dataset_dir,
            crop_size: int = 96,
            upscale_factor: int = 4,
            random_resize: bool = False,
    ):
        super(TrainDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

        self.crop_size = crop_size
        self.crop_patch = RandomCrop(crop_size)
        self.hr_to_lr = Resize(crop_size // upscale_factor, InterpolationMode.BICUBIC)

        self.flip = Compose(
                [RandomHorizontalFlip(p=0.5),
                 RandomVerticalFlip(p=0.5)]
        )
        self.random_resize = random_resize

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])

        if self.random_resize and np.random.rand() < 0.5 :
            hr_image = self.random_resize_image(hr_image)
        hr_image = self.crop_patch(hr_image)
        hr_image = self.patch_augmentation(hr_image)
        lr_image = self.hr_to_lr(hr_image)

        hr = to_tensor(hr_image)
        lr = to_tensor(lr_image)
        return lr, hr
    
    def random_resize_image(self, hr_image) :
        resize_factor = np.random.choice([2, 3, 4])
        W, H = hr_image.size
        h, w = H // resize_factor, W // resize_factor
        if min(h, w) < self.crop_size :
            return hr_image
        else :
            return resize(hr_image, (h, w))

    def patch_augmentation(self, hr_patch) :
        hr_patch = self.flip(hr_patch)
        if np.random.rand() < 0.5 :
            hr_patch = rotate(hr_patch, 90)
        return hr_patch


    def __len__(self):
        return len(self.image_filenames)

class ValDataset(Dataset):
    def __init__(
            self,
            dataset_dir,
            upscale_factor: int = 4,
    ):
        super(ValDataset, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        W, H = hr_image.size
        H_, W_ = (H - H % self.upscale_factor), (W - W % self.upscale_factor)
        h, w = H_ // self.upscale_factor, W_ // self.upscale_factor
        hr_image = CenterCrop((H_, W_))(hr_image)
        lr_image = Resize((h, w), InterpolationMode.BICUBIC)(hr_image)

        hr = to_tensor(hr_image)
        lr = to_tensor(lr_image)
        return lr, hr

    def __len__(self):
        return len(self.image_filenames)

class TestDataset(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDataset, self).__init__()
        self.upscale_factor = upscale_factor

        lr_path = os.path.join(dataset_dir, 'LR')
        hr_path = os.path.join(dataset_dir, 'HR')
        self.lr_filenames = [os.path.join(lr_path, x) for x in sorted(listdir(lr_path)) if is_image_file(x)]
        self.hr_filenames = [os.path.join(hr_path, x) for x in sorted(listdir(hr_path)) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        hr_image = Image.open(self.hr_filenames[index])
        W, H = hr_image.size
        H_, W_ = (H - H % self.upscale_factor), (W - W % self.upscale_factor)
        h, w = H_ // self.upscale_factor, W_ // self.upscale_factor
        hr_image = CenterCrop((H_, W_))(hr_image)
        lr_image = Resize((h, w), InterpolationMode.BICUBIC)(hr_image)
        bicubic_sr_image = Resize((H_, W_), InterpolationMode.BICUBIC)(lr_image)

        lr, hr, bicubic_sr = to_tensor(lr_image), to_tensor(hr_image), to_tensor(bicubic_sr_image)
        return image_name, lr, hr, bicubic_sr

    def __len__(self):
        return len(self.lr_filenames)

