import os
from os.path import exists, join


direc = "RaFD/"

files = [filename for filename in os.listdir(direc)]

for file in files:
    new_filename = file.replace('Rafd', '')
    os.rename(join(direc, file), join(direc, new_filename))
