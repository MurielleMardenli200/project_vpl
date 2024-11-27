import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from skimage import io
import numpy as np
from PIL import Image
import os
from stimupy.noises.whites import white
from stimupy.noises.naturals import one_over_f
from typing import Callable 
import random
from torchvision.transforms import functional as F


from paths import DS_DIR


def noise_generator(noise_type,visual_size,intensity_range,exponent = 1):
    if noise_type == "white":
        noise = white(
            visual_size=visual_size,
            ppd=None,
            shape=visual_size,
            intensity_range=intensity_range,
            pseudo_noise=False,
        )["img"]
        
    elif noise_type == "one_over_f":
        noise = one_over_f(
            visual_size=visual_size,
            ppd=None,
            shape=visual_size,
            exponent=exponent,
            intensity_range=intensity_range,
            pseudo_noise=False,
        )["img"]

    elif noise_type == 'None':

        noise = np.zeros(visual_size)
    else:
        raise ValueError("Unsupported noise type")

    return torch.from_numpy(noise).unsqueeze(0).float()
    

def add_noise(tensor: torch.Tensor,noise_type: str,intensity_range: tuple) -> torch.Tensor:
        
        noise = noise_generator(noise_type=noise_type,visual_size=tensor.shape[1:],intensity_range=intensity_range)

        imagenet_mean=0.449

        tensor = (tensor - tensor.mean()) * 0.2 + imagenet_mean

        tensor = tensor + noise

        tensor = ((tensor -  tensor.min()) / (tensor.max() - tensor.min()))

        tensor = torch.cat([tensor,tensor,tensor])

        return tensor


def create_transform(noise_type, intensity_range):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(),
        transforms.Lambda(lambda x: apply_noise(x, noise_type, intensity_range)),
    ])
    return transform



def create_circular_mask(h, w, radius=None):
    center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask

def apply_aperature(tensor: torch.Tensor, type: str, radius: int) -> torch.Tensor:
    _, h, w = tensor.shape

    if type == 'circular_gaussian':
        mask = create_circular_mask(h, w, radius)
        grey_background = 0.5
        background_color = grey_background
        center = (w // 2, h // 2)
        Y, X = np.ogrid[:h, :w]
        dist_from_center = torch.tensor(np.sqrt((X - center[0])**2 + (Y - center[1])**2), dtype=torch.float32)
        sigma_gaussian = 50
        gaussian_aperture = (1 - torch.exp(-((dist_from_center / sigma_gaussian) ** 2) / 2)) * 1

        tensor = torch.tensor(tensor, dtype=torch.float32)

        for channel in range(3):
            tensor[channel][~mask] = background_color
            tensor[channel][mask] = tensor[channel][mask] * (1 - gaussian_aperture[mask]) + background_color * gaussian_aperture[mask]
        
        #tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    return tensor


def zoom_in(tensor: torch.Tensor, zoom_factor: int, center=None):
    """
    Zooms into an image at the specified center and zoom factor.
    """
    _,width, height = tensor.shape
    if center is None:
        center = (width // 2, height // 2)

    # Calculating the size of the cropped area
    new_width = width // zoom_factor
    new_height = height // zoom_factor

    # Calculating the coordinates for cropping
    left = center[0] - new_width // 2
    top = center[1] - new_height // 2
    right = center[0] + new_width // 2
    bottom = center[1] + new_height // 2

    # Crop and resize
    tensor = F.crop(tensor, top, left, new_height, new_width)
    tensor = F.resize(tensor, (height, width))
    return tensor

def place_on_background(tensor: torch.Tensor, bg_size: tuple, position: tuple) -> torch.Tensor:
    background = torch.full((3, *bg_size), 0.5) 
    x, y = position
    _, h, w = tensor.shape
    background[:, y:y+h, x:x+w] = tensor
    return background

def create_transform_aperature(noise_type, intensity_range, type, radius, bg_size, position, zoom_factor, rotation=None, horizontal_flip=False, vertical_flip=False):

    def apply_noise_wrapper(x):
        return apply_noise(x, noise_type, intensity_range)

    def zoom_in_wrapper(img):
        return zoom_in(img, zoom_factor=zoom_factor)

    def apply_aperature_wrapper(img):
        return apply_aperature(img, type, radius)

    def place_on_background_wrapper(img):
        return place_on_background(img, bg_size, position)

    transform_list = [
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Grayscale()
    ]
    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip())
    if rotation is not None:
        transform_list.append(transforms.RandomRotation(degrees=rotation))

    transform_list.extend([
        transforms.Lambda(apply_noise_wrapper),
        transforms.Lambda(zoom_in_wrapper),
        transforms.Lambda(apply_aperature_wrapper),
        transforms.Lambda(place_on_background_wrapper),
        transforms.Resize((224, 224))
    ])

    return transforms.Compose(transform_list)

def generate_test_transforms(config):
    
    transform = create_transform_aperature(
        noise_type=config.get('noise_type'),
        intensity_range=config.get('intensity_range'),
        type=config.get('type'),
        radius=config.get('radius'),
        bg_size=config.get('bg_size'),
        position=config.get('position'),
        zoom_factor=config.get('zoom_factor'),
        rotation=config.get('rotation'),
        horizontal_flip=config.get('horizontal_flip'),
        vertical_flip=config.get('vertical_flip')
    )
    
    return transform


def apply_noise(img, noise_type, intensity_range):
    img = add_noise(img, noise_type, intensity_range)
    return img


class NSD(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.num_images = len(self.file_list)
        self.transforms = transforms
        
    def __len__(self):
        return self.num_images

    def create_label(self, filename):
        label = int("".join(filter(str.isdigit, filename)))
        return label
    
    def apply_transform(self, img):
        if self.transforms and img.mode == 'RGB':
            return self.transforms(img)
        return img

    def load_images(self,file_name):
        img = Image.open(os.path.join(self.data_dir, file_name)).convert('RGB') 
        return img

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        reference_filename = self.file_list[idx]

        reference_image = self.load_images(reference_filename)
        
        label = self.create_label(reference_filename)

        img1 = self.apply_transform(reference_image)
        
        label2 = label
       
        return img1, label2

class COCO(Dataset):
    def __init__(self, data_dir, transforms=None,same_pair_probability=0.5,same_reference=False, same_not_rand=False, ref_state = 100, ref_other = 50, idx_ref=None, idx_other=None):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.num_images = len(self.file_list)
        self.transforms = transforms
        self.same_pair = None
        self.same_pair_probability = same_pair_probability
        self.same_reference = same_reference
        self.same_not_rand = same_not_rand
        self.ref_state = ref_state
        self.ref_other = ref_other
        self.idx_ref = idx_ref
        self.idx_other = idx_other
        
    def __len__(self):
        return self.num_images

    def create_label(self, filename):
        label = int("".join(filter(str.isdigit, filename)))
        return label
    
    def _same_different(self):
        self.same_pair = random.random() < self.same_pair_probability

    def apply_transform(self, img):
        if self.transforms and img.mode == 'RGB':
            return self.transforms(img)
        return img


    def two_reference_task(self):
        rng1 = np.random.RandomState(self.ref_state)
        rng2 = np.random.RandomState(self.ref_other)

        idx = rng1.randint(0, self.num_images)
        reference_filename = self.file_list[idx]

        other_idx = rng2.randint(0, self.num_images)
        other_filename = self.file_list[other_idx]

        return reference_filename, other_filename
    
    def same_different_task(self,idx):

        image_filename = self.file_list[idx]

        random_idx = random.choice(range(self.num_images))
        while random_idx == idx:

            random_idx = random.choice(range(self.num_images))

        random_filename = self.file_list[random_idx]

        return image_filename,random_filename
    
    def same_different_not_rand(self):

        image_filename = self.file_list[self.idx_ref]

        other_filename = self.file_list[self.idx_other]

        return image_filename,other_filename
    
    def load_images(self,file_name):
        img = Image.open(os.path.join(self.data_dir, file_name)).convert('RGB') 
        return img


    def __getitem__(self, idx):

        self._same_different()

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.same_reference:

            reference_filename, other_filename = self.two_reference_task()

        elif self.same_not_rand:

            reference_filename, other_filename = self.same_different_not_rand()

        else:
            
            reference_filename, other_filename = self.same_different_task(idx)

        reference_image = self.load_images(reference_filename)

        other_image = self.load_images(other_filename)
        
        label = self.create_label(reference_filename)

        label2 = self.create_label(other_filename)

        if self.same_pair:
            img1 = self.apply_transform(reference_image)
            img2 = self.apply_transform(reference_image)
            label1 = label
            label2 = label
        else:
            img1 = self.apply_transform(reference_image)
            img2 = self.apply_transform(other_image)

            label1 = label

            label2 = label2

        return img1, img2, label1, label2, self.same_pair



def get_data_loaders(data_dir, transform, batch_size=128, same_pair_probability=0.5, same_reference = False, same_not_rand = False, ref_state = 100, ref_other = 50, idx_ref=None, idx_other=None):

    data_dir = data_dir

    transform = transform

    if same_not_rand:
        shuffle = False
    else:
        shuffle = True

    train_dir = os.path.join(data_dir, 'shared1000')

    val_dir = os.path.join(data_dir, 'shared1000')

    train_dataset = COCO(train_dir, transforms=transform,same_pair_probability=same_pair_probability,same_reference=same_reference, same_not_rand=same_not_rand, ref_state = ref_state,ref_other=ref_other, idx_ref=idx_ref, idx_other=idx_other)

    test_dataset = COCO(val_dir, transforms=transform,same_pair_probability=same_pair_probability,same_reference=same_reference, same_not_rand=same_not_rand, ref_state = ref_state,ref_other=ref_other, idx_ref=idx_ref, idx_other=idx_other)

    batch_size_train = batch_size

    batch_size_test = batch_size
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=shuffle,drop_last=True,num_workers=0) #change to 8

    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=shuffle,drop_last=True,num_workers=0) # change to 8

    return train_loader, test_loader
