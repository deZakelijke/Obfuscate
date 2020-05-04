import os
from os.path import exists, join


direc = "crowd_mugshots/"

files = [filename for filename in os.listdir(direc)]

for file in files:
    if '_p0_y0_r0_p0_y0_r0' in file:
        new_filename = file.replace('_p0_y0_r0_p0_y0_r0', '_p0_y0_r0')
        os.rename(join(direc, file), join(direc, new_filename))
