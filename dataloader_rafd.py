import os
import csv
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

#### REAL DATA ####

# Viewpoint encodings
VL2 = "000"  # -90 (right side visible)
VL1 = "045"  # -45 (right side visible)
VN  = "090"  # 0   (frontal view)
VR1 = "135"  # 45  (left side visible)
VR2 = "180"  # 90  (left side visible) 

# Expression
HAPPY        = "happy"
CONTEMPTUOUS = "contemptuous"
DISGUSTED    = "disgusted"
FEARFUL      = "fearful"
NEUTRAL      = "neutral"
ANGRY        = "angry"
SURPRISED    = "surprised"
SAD          = "sad"

# Ethnicity of model
KID       = "Kid"
CAUCASIAN = "Caucasian"
MOROCCAN  = "Moroccan"

# Gazes
RIGHT =   "right"
FRONTAL = "frontal"
LEFT =    "left"


TMP_FILENAME = "temporary_sources.csv"

class ModeRafd():
    """
    Class to wrap the parameters used in the split.
    The parameters given are the ones used in the *targets*!
    Everything else is used as sources.
    """
    def __init__(self, number_of_targets, viewpoint, ethnicity, expression, gaze):
        self.number_of_targets = number_of_targets
        self.viewpoint = viewpoint
        self.ethnicity = ethnicity
        self.expression = expression
        self.gaze = gaze


def get_mode(mode, number_of_targets=100):
    print("Mode is", mode)
    # 100 targets without augmentations
    if mode == 1:
        return ModeRafd(number_of_targets=number_of_targets,
                    expression=[NEUTRAL],
                    viewpoint=[VN],
                    ethnicity=[CAUCASIAN],
                    gaze=[FRONTAL])


class DatasetCreator:
    source_files = None
    target_files = None
    target_classes = None
    mode = None
    df = None # panda eyes dataframe
    load_images = None

    def __init__(self, source_files_location, augmented_location, mode,
                 number_of_targets=100, seed=42, load_saved_sources=False,
                 load_images=False):
        np.random.seed(seed)
        self.load_images = load_images

        # If user provides mode we do not need to construct one of the default objects
        if type(mode) == ModeRafd:
            self.mode = mode
        else:
            self.mode = get_mode(mode, number_of_targets)

        print(source_files_location)
        
        source_files = os.listdir(source_files_location)
        source_files = list(filter(lambda x: ".DS_Store" not in x, source_files)) # remove .DS_Store
        source_files.sort()
        if augmented_location is not None:
            augmented_files = os.listdir(augmented_location)
            augmented_files = list(filter(lambda x: ".DS_Store" not in x, augmented_files)) # remove .DS_Store
            augmented_files.sort()
        else:
            augmented_files = None

        self.df = self.get_dataframe(source_files_location, source_files, augmented_location, augmented_files)

        # split the dataset using the provided mode
        self.split_data()

        if load_saved_sources:
            print('Loading source files from previous run')
            with open(TMP_FILENAME, 'r') as f:
                self.source_files = f.read().splitlines()

    # create dataframe with paths to the files
    def get_dataframe(self, source_files_location, source_files, augmented_location, augmented_files):
        file_matrix_source = np.array([[os.path.join(source_files_location, source_files[i])] + fn.split(".")[0].split("_") for i, fn in enumerate(source_files)])

        if augmented_location is not None:
            file_matrix_augmented = np.array([[os.path.join(augmented_location, augmented_files[i])] + fn.split(".")[0].split("_") for i, fn in enumerate(augmented_files)])
            file_matrix = np.concatenate((file_matrix_source, file_matrix_augmented))
        else:
            file_matrix = file_matrix_source
        columns = ["filename", "viewpoint", "person_ID", "ethnicity", "gender", "expression", "gaze"]
        df = pd.DataFrame(file_matrix, columns=columns)
        return df

    def split_data(self):
        target_df = self.df.copy()

        # Dataset variations
        target_df = target_df[target_df["expression"].isin(self.mode.expression)]
        target_df = target_df[target_df["viewpoint"].isin(self.mode.viewpoint)]
        target_df = target_df[target_df["ethnicity"].isin(self.mode.ethnicity)]
        target_df = target_df[target_df["gaze"].isin(self.mode.gaze)]

        unique_ids = target_df["person_ID"].unique()
        np.random.shuffle(unique_ids)
        target_actors = unique_ids[:self.mode.number_of_targets]
        target_df = target_df[target_df["person_ID"].isin(target_actors)]
        self.df = self.df.drop(index=target_df.index, errors='ignore')
        self.target_df = target_df
        self.source_files = self.df["filename"].values
        self.target_files = self.target_df["filename"].values
        self.target_classes = set(self.target_df["person_ID"].astype(int).values)

    def get_source_dataset(self):
        return RaFDDataset(self.source_files, load_images=self.load_images)

    def get_target_dataset(self):
        return RaFDDataset(self.target_files, self.target_classes, load_images=self.load_images), self.target_classes


class RaFDDataset(Dataset):
    file_locations = None
    classes_set = None
    load_images = None

    def __init__(self, file_locations, classes_set=None, load_images=False):
        self.file_locations = file_locations
        self.classes_set = classes_set
        self.load_images = load_images

    def __len__(self):
        return len(self.file_locations)

    def __getitem__(self, idx):
        image_filename = self.file_locations[idx]
        if self.load_images:
            image = Image.open(image_filename)
            image = transforms.ToTensor()(image)
            image.requres_grad = False
            return {"image" : image, "filename" : image_filename}
        # because of the batches, the values are arrays!
        return {"filename" : image_filename}

def get_dataloader(source_files, augmented_files=None, mode=1,
                   number_of_targets=100, batch_size=1, shuffle=True,
                   num_workers=1, seed=42, load_saved_sources=False,
                   load_images=False):
    """
    Returns a dataloader to be used for classification.
    Usage:
    source_dataloader, target_dataloader, target_classes = get_dataloader(<path>, <augmented_path>, mode=[int | m: Mode])
    # where mode is either an int, using one of the default modes or a Mode object
    for file in source_dataloader:
        # file["filename"][0] because we return a batch with size 1
        # the split on the "/" is because we append the entire path to make sure
        # that the system can find the image independent of whether it was original
        # or an augmented version
        id = file["filename"][0].split("/")[-1].split("_")[0]
        image = file["image"][0] # torch tensor of WxHxC
        should_find_match = id in target_classes
        predicted_id = predict(file)
        if should_find_match:
            correct = True if predicted_id == id else False
        else:
            correct = False if predicted_id != -1 else True

    :param source_files: path to the full dataset
    :param batch_size: size of the batch, default 1
    :param shuffle: if the data should be loaded in order or not
    :param mode: which mode to use to make the source/target split
    :return: source dataloader, target dataloader, set of unique classes (str) in target set
    """
    dataset_creator = DatasetCreator(source_files, augmented_files, mode,
                                     number_of_targets=number_of_targets, seed=seed,
                                     load_saved_sources=load_saved_sources,
                                     load_images=load_images)
    source_dataset = dataset_creator.get_source_dataset()
    target_dataset, target_classes = dataset_creator.get_target_dataset()

    source_dataloader = DataLoader(source_dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)

    target_dataloader = DataLoader(target_dataset, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers)

    return source_dataloader, target_dataloader, target_classes


def save_sources(source_dataloader, target_classes=None):
    print("going to write", len(source_dataloader), "lines")
    with open(TMP_FILENAME, 'w') as f:
        w = csv.writer(f)
        for l in source_dataloader:
            if target_classes is not None:
                if int(l["filename"][0].split("/")[-1].split("_")[0]) not in target_classes:
                    continue
            w.writerow(l["filename"])
    return

def remove_sources():
    os.remove(TMP_FILENAME)
