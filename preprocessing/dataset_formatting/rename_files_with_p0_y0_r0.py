import os
from os.path import exists, join


dir1 = "crowd_mugshots/"
dir2 = "target_mugshots/"
dir3 = "unused_target_mugshots/"

dirs = [dir1, dir2, dir3]

all_files = [[filename for filename in os.listdir(directory)] for directory in dirs]

for directory, files in zip(dirs, all_files):
    for file in files:
        new_filename = file[:-4] + "_p0_y0_r0.png"
        os.rename(join(directory, file), join(directory, new_filename))
