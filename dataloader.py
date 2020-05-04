import os
import csv
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

#### REAL DATA ####

# Viewpoint encodings
VL3 = "080"  # -45
VL2 = "130"  # -30
VL1 = "140"  # -15
VN  = "051"  # 0
VR1 = "050"  # 15
VR2 = "041"  # 30
VR3 = "190"  # 45

# Illumination encodings
IL   = "02"  # Left
IF   = "07"  # Front
IR   = "12"  # Right
IALL = "17"  # All angles

# Expression encodings
NEUTRAL = "0"
HAPPY   = "1"
WILD    = "2"


#### FAKE DATA ####

# Yaw
Y0   = "y0"
Y15  = "y15"
Y_15 = "y-15"
Y30  = "y30"
Y_30 = "y-30"
Y45  = "y45"
Y_45 = "y-45"
Y60  = "y60"
Y_60 = "y-60"

# Pitch
P0   = "p0"
P15  = "p15"
P_15 = "p-15"
P30  = "p30"
P_30 = "p-30"
P45  = "p45"
P_45 = "p-45"
P60  = "p60"
P_60 = "p-60"

# Roll
R0   = "r0"
R15  = "r15"
R_15 = "r-15"

# Illumination direction
ILN = "il0"  # no light change
ILL = "il1"  # left lighting
ILR = "il2"  # right lighting
ILF = "il3"  # front lighting

# Illumination intensity
INN = "in0"   # low intensity
IN1 = "in10"  # low intensity
IN3 = "in35"  # medium intensity
IN6 = "in60"  # clipping intensity

TMP_FILENAME = "temporary_sources.csv"


class Mode():
    """
    Class to wrap the parameters used in the split.
    The parameters given are the ones used in the *targets*!
    Everything else is used as sources.
    """

    def __init__(self, number_of_targets, expression, viewpoint,
                 illumination, pitch, yaw, roll, illum_change, illum_intensity, glasses):
        self.number_of_targets = number_of_targets
        self.expression = expression
        self.viewpoint = viewpoint
        self.illumination = illumination
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.illum_change = illum_change
        self.illum_intensity = illum_intensity
        self.glasses = glasses


def get_mode(mode, number_of_targets=100):
    print("Mode is", mode)
    # 100 targets without augmentations
    if mode == 0:
        return Mode(number_of_targets=0,
                    expression=[],
                    viewpoint=[],
                    illumination=[],
                    pitch=[],
                    yaw=[],
                    roll=[],
                    illum_change=[],
                    illum_intensity=[],
                    glasses=[])
    elif mode == 1:
        return Mode(number_of_targets=number_of_targets,
                    expression=[NEUTRAL],
                    viewpoint=[VN],
                    illumination=[IALL],
                    pitch=[P0],
                    yaw=[Y0],
                    roll=[R0],
                    illum_change=[ILN],
                    illum_intensity=[INN],
                    glasses=[])
    # 100 targets with viewpoint augmentations
    elif mode == 2:
        return Mode(number_of_targets=number_of_targets,
                    expression=[NEUTRAL],
                    viewpoint=[VN],
                    illumination=[IALL],
                    pitch=[P0, P_15, P15],
                    yaw=[Y0, Y_15, Y15],
                    roll=[R0],
                    illum_change=[ILN],
                    illum_intensity=[INN],
                    glasses=[])
    # 100 targets with illumination augmentations
    elif mode == 3:
        return Mode(number_of_targets=number_of_targets,
                    expression=[NEUTRAL],
                    viewpoint=[VN],
                    illumination=[IALL],
                    pitch=[P0],
                    yaw=[Y0],
                    roll=[R0],
                    illum_change=[ILN, ILL, ILR, ILF],
                    illum_intensity=[INN, IN1, IN3, IN6],
                    glasses=[])
    # 100 targets with viewpoint and illumination augmentations
    elif mode == 4:
        return Mode(number_of_targets=number_of_targets,
                    expression=[NEUTRAL],
                    viewpoint=[VN],
                    illumination=[IALL],
                    pitch=[P0, P_15, P15],
                    yaw=[Y0, Y_15, Y15],
                    roll=[R0],
                    illum_change=[ILN, ILL, ILR, ILF],
                    illum_intensity=[INN, IN1, IN3, IN6],
                    glasses=[])
    # 100 targets without augmentations, 5 mugshots,
    # 1 viewpoint and 1 light change extra
    if mode == 5:
        return Mode(number_of_targets=number_of_targets,
                    expression=[NEUTRAL],
                    viewpoint=[VN, VL1],
                    illumination=[IALL, IR],
                    pitch=[P0],
                    yaw=[Y0],
                    roll=[R0],
                    illum_change=[ILN],
                    illum_intensity=[INN],
                    glasses=[])
    else:
        return None


class DatasetCreator:
    source_files = None
    target_files = None
    target_classes = None
    mode = None
    combined_df = None  # panda eyes dataframe
    source_df = None
    load_images = None

    def __init__(self, source_files_location, augmented_location, mode,
                 number_of_targets=100, seed=42, load_saved_sources=False,
                 load_images=False):
        np.random.seed(seed)
        self.load_images = load_images

        # If user provides mode we do not need to construct one of the default objects
        if type(mode) == Mode:
            self.mode = mode
        else:
            self.mode = get_mode(mode, number_of_targets)

        print(source_files_location)

        source_files = os.listdir(source_files_location)
        source_files = list(filter(lambda x: ".DS_Store" not in x, source_files))  # remove .DS_Store
        source_files.sort()
        if augmented_location is not None:
            augmented_files = os.listdir(augmented_location)
            augmented_files = list(filter(lambda x: ".DS_Store" not in x, augmented_files))  # remove .DS_Store
            augmented_files.sort()
        else:
            augmented_files = None

        self.combined_df, self.source_df = self.get_dataframe(source_files_location, source_files, augmented_location, augmented_files)

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
        columns = ["filename", "id", "session", "pose", "illumination", "expression", "pitch", "yaw", "roll", "il", "in", "glasses"]
        source_df = pd.DataFrame(file_matrix_source, columns=columns)
        combined_df = pd.DataFrame(file_matrix, columns=columns)
        return combined_df, source_df

    def split_data(self):
        target_df = self.combined_df.copy()

        # Dataset variations
        target_df = target_df[target_df["expression"].isin(self.mode.expression)]
        target_df = target_df[target_df["pose"].isin(self.mode.viewpoint)]
        target_df = target_df[target_df["illumination"].isin(self.mode.illumination)]

        # Pose augmentations
        target_df = target_df[target_df["pitch"].isin(self.mode.pitch)]
        target_df = target_df[target_df["yaw"].isin(self.mode.yaw)]
        target_df = target_df[target_df["roll"].isin(self.mode.roll)]

        # Illumination augmentations
        target_df = target_df[target_df["il"].isin(self.mode.illum_change)]
        target_df = target_df[target_df["in"].isin(self.mode.illum_intensity)]

        if len(target_df) == 0:
            self.target_files = []
            self.target_classes = set()
            self.source_files = self.source_df["filename"].values
            return

        unique_ids = target_df["id"].unique()
        np.random.shuffle(unique_ids)
        target_actors = unique_ids[:self.mode.number_of_targets]
        target_df = target_df[target_df["id"].isin(target_actors)]
        # print("length before duplicate removal", len(target_df))
        target_df = target_df.drop_duplicates(subset=["id", "pose", "illumination", "expression", "pitch", "yaw", "roll", "il", "in", "glasses"])
        self.target_df = target_df
        # print("length after duplicate removal", len(target_df))
        # drop the targets from the source
        self.source_df = self.source_df.drop(index=target_df.index, errors='ignore')
        self.source_files = self.source_df["filename"].values
        self.target_files = self.target_df["filename"].values
        self.target_classes = set(self.target_df["id"].astype(int).values)

    def get_source_dataset(self):
        return MultiPieDataset(self.source_files, load_images=self.load_images)

    def get_target_dataset(self):
        return MultiPieDataset(self.target_files, self.target_classes, load_images=self.load_images), self.target_classes


class MultiPieDataset(Dataset):
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
            return {"image": image, "filename": image_filename}
        # because of the batches, the values are arrays!
        return {"filename": image_filename}


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
    #            w.writerows([f for f in l["filename"]])
    return


def remove_sources():
    os.remove(TMP_FILENAME)
