import os
from os.path import exists, join

dir1 = 'augmented_illumination_preprocessed/'
dir2 = 'facenet_preprocessed/'
dir3 = 'MultiePIE_all_data/'
dir4 = 'target_mugshots_augmented_illumination_multipie/renders/'
dir5 = 'target_mugshots/'

dirs = [dir1, dir2, dir3, dir4, dir5]

all_files = [[filename for filename in os.listdir(directory) if os.path.splitext(filename)[1] == '.png'] for directory in dirs]

for directory, files in zip(dirs, all_files):
    for file in files:
        new_filename = file[:-4] + "_0.png"
        os.rename(join(directory, file), join(directory, new_filename))
