import os
from shutil import copyfile

basedir = "/home/douwe/Documents/target_mugshots/"
newdir = "/home/douwe/Documents/openface_target_mugshots/"

files = os.listdir(basedir)

if not os.path.isdir(newdir):
    os.mkdir(newdir)

for file in files:
    person_id = file.split("_")[0]
    if not os.path.isdir(person_id):
        os.mkdir(newdir + person_id)
    copyfile(basedir + file, newdir + "/" + person_id + "/" +  file)

