import torch

from os import listdir
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils


class FaceLoader(Dataset):

    def __init__(self, nr_images, data_dir="MultiPIE_sample/"):
        "Format for folders is: id_session_angle_illumination"

        self.nr_images = nr_images
        self.data_dir = data_dir
        self.image_folder = listdir(data_dir)

        if nr_images > len(self.image_folder):
            raise IndexError("Not enough images in dataset")

    def __len__(self):
        return self.nr_images

    def __getitem__(self, index):
        image_name = self.image_folder[index]
        image = Image.open(f"{self.data_dir}{image_name}")
        image = transforms.ToTensor()(image).float()
        image.requires_grad = False
        return {"image": image, "label" : image_name}
        
