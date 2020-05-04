import os
import shutil
from os.path import join, isdir, exists

base_path = 'MultiPIE/Frames/'
target_dir = 'MultiPIE_all/'
folders = [folder for folder in os.listdir(base_path) if isdir(join(base_path, folder))]

if not exists(target_dir):
    os.makedirs(target_dir)


for f_id, f in enumerate(folders):
    if f_id % 1000 == 0:
        print("{}/{}".format(f_id, len(folders)))
    for i, img_name in enumerate(os.listdir(join(base_path, f))):
        shutil.copyfile(join(base_path, f, img_name), join(target_dir, f + '_' + str(i) + '.png'))
