import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
from progressbar import ProgressBar
from mtcnn import MTCNN
import argparse

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
    device=device
)


parser = argparse.ArgumentParser(description='preprocessing for Facenet')
parser.add_argument('--images_loc', type=str, default='/home/douwe/Downloads/multi_pie/data/', help='Location of images to be preprocessed')
parser.add_argument('--save_loc', type=str, default='/home/douwe/Downloads/multi_pie/processed/', help='Location of preprocessed images to be saved')

if __name__ == '__main__':
    args = parser.parse_args()

    trans = transforms.Compose([transforms.Resize(1024)])
    dataset = ImageFolderWithPaths(args.images_loc, transform=trans)
    loader = DataLoader(dataset, collate_fn=lambda x: x[0], num_workers=4, batch_size=1)

    bar = ProgressBar(max_value=len(loader))
    not_found = []
    probs = []

    print("ye")
    for input_image, _, image_name in loader:
        if not os.path.exists(args.save_loc + image_name.split('/')[-1]):
            x_aligned, prob = mtcnn(input_image, return_prob=True, remove_eyes = False)

            if x_aligned is None:
                not_found.append(image_name)
                probs.append(prob)
            else:
                utils.save_image(x_aligned, args.save_loc + image_name.split('/')[-1])

        bar += 1

    print(not_found)
    with open('nout_found.txt', 'w') as f:
        for item in not_found:
            f.write("%s\n" % item)
