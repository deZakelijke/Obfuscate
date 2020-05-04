import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

def get_image(filename):
    image = Image.open(filename)
    image = transforms.ToTensor()(image)
    image.requres_grad = False
    return image


class TrainDataset(Dataset):
    triplets = None

    def __init__(self, triplets, load_images=False, seed=42):
        self.load_images = load_images
        np.random.seed(seed)
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        (anchor, positive, negative) = self.triplets[idx]

        anchor_dict = {"filename": anchor}
        positive_dict = {"filename": positive}
        negative_dict = {"filename": negative}

        if self.load_images:
            anchor_dict["image"] = get_image(anchor)
            positive_dict["image"] = get_image(positive)
            negative_dict["image"] = get_image(negative)

        # because of the batches, the values are arrays!
        return anchor_dict, positive_dict, negative_dict


def get_train_dataloader(triplets, batch_size=1, shuffle=True,
                         num_workers=1, seed=42,
                         load_images=False):
    target_dataset = TrainDataset(triplets, load_images=load_images, seed=seed)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers)

    return target_dataloader



class EasyDataset(Dataset):
    def __init__(self, source_files_location, augmented_location, load_images=False, seed=42):
        self.load_images = load_images
        np.random.seed(seed)

        source_files = os.listdir(source_files_location)
        source_files = list(filter(lambda x: ".DS_Store" not in x, source_files))  # remove .DS_Store
        source_files.sort()
        if augmented_location is not None:
            augmented_files = os.listdir(augmented_location)
            augmented_files = list(filter(lambda x: ".DS_Store" not in x, augmented_files))  # remove .DS_Store
            augmented_files.sort()
        else:
            augmented_files = None

        self.df = self.get_dataframe(source_files_location, source_files, augmented_location, augmented_files)

    def get_dataframe(self, source_files_location, source_files, augmented_location, augmented_files):
        file_matrix_source = np.array(
            [[os.path.join(source_files_location, source_files[i])] + fn.split(".")[0].split("_") for i, fn in
             enumerate(source_files)])
        if augmented_location is not None:
            file_matrix_augmented = np.array(
                [[os.path.join(augmented_location, augmented_files[i])] + fn.split(".")[0].split("_") for i, fn in
                 enumerate(augmented_files)])
            file_matrix = np.concatenate((file_matrix_source, file_matrix_augmented))
        else:
            file_matrix = file_matrix_source
        columns = ["filename", "id", "session", "pose", "illumination", "expression", "pitch", "yaw", "roll", "il", "in", "glasses"]
        # NOTE THAT THE SOURCE NOW CONTAINS BOTH ORIGINAL AND AUGMENTED IMAGES!
#        file_matrix = file_matrix[:1000] # TODO: REMOVE PL0X
        df = pd.DataFrame(file_matrix, columns=columns)
        return df

    def get_positive(self, filename):
        target_label = filename.split("/")[-1].split("_")[0]
        matches = self.df.loc[(self.df["filename"] != filename) & (self.df["id"] == target_label)]  # .sample(n=1)
        if len(matches) > 0:
            return matches["filename"].values
        else:
            raise Exception("WTF IS GIONG ON BOII")

    def get_negative(self, filename):
        target_label = filename.split("/")[-1].split("_")[0]
        matches = self.df.loc[(self.df["filename"] != filename) & (self.df["id"] != target_label)].sample(n=3000, replace=True)
        if len(matches) > 0:
            return matches["filename"].values
        else:
            raise Exception("WTF IS GIONG ON BOII")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_filename = self.df.iloc[idx]["filename"]
        image_dict = {"filename": image_filename}

        if self.load_images:
            image_dict["image"] = get_image(image_filename)

        return image_dict

def get_easy_dataloader(source_files, augmented_files=None, batch_size=1, shuffle=True,
                        num_workers=1, seed=42,
                        load_images=False):
    target_dataset = EasyDataset(source_files, augmented_files, load_images=load_images, seed=seed)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers)

    return target_dataloader, target_dataset # needed for that boi negative and positive


