import cv2
import dlib
import torch
import numpy as np
from mtcnn import MTCNN
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
            device=device)

image = np.array(Image.open("/home/douwe/Downloads/twopeople.jpg"))
image = np.array(Image.open("/home/douwe/Downloads/ali.jpg"))


x_aligned, prob, boxes = mtcnn(Image.fromarray(image), return_prob=True, remove_eyes=False, keep_all=False)

print(x_aligned.shape)