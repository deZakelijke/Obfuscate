import os
from os.path import exists, join

base_dir = "MultiPIE_all/"
target_dir = "target_mugshots/"
n_persons = 346 # max 346

if not exists(target_dir):
    os.makedirs(target_dir)

for person in range(1, n_persons+1):
    for session in range(1,5):
        person_str = (3-len(str(person))) * '0' + str(person)
        filename = "{}_0{}_051_17_0.png".format(person_str, session)
        try:
            os.rename(join(base_dir, filename), join(target_dir, filename))
        except:
            print("Person {} does not exist! Skipping...".format(person_str))
