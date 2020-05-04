"""
This file looks at a folder with a lot of target mugshots with frontal
viewpoint and complete illumination of neutral expressions, and filters
out all personIDs above 100, and makes sure there is only one image per
person (not multiple sessions).
"""

import os
from os.path import exists, join

base_dir = "target_mugshots/"
target_dir = "unused_target_mugshots/"
cutoff_person_id = 101 # every personID from this point on will be moved
n_persons = 346
n_sessions = 4

if not exists(target_dir):
    os.makedirs(target_dir)
    
# first move all files of people with an ID above some threshold
for person in range(cutoff_person_id, n_persons+1):
    found_file = False
    for session in range(1, n_sessions + 1):
        person_str = (3-len(str(person))) * '0' + str(person)
        filename = "{}_0{}_051_17_0.png".format(person_str, session)
        if exists(join(base_dir, filename)):
            try:
                os.rename(join(base_dir, filename), join(target_dir, filename))
            except:
                print("ERROR: cannot move file {}".format(join(base_dir, filename)))

# then move files that appear twice for one person (multiple sessions)
for person in range(1, n_persons+1):
    found_file = False
    for session in range(1, n_sessions + 1):
        person_str = (3-len(str(person))) * '0' + str(person)
        filename = "{}_0{}_051_17_0.png".format(person_str, session)
        if exists(join(base_dir, filename)):
            if not found_file:
                found_file = True
                continue
            else:                
                try:
                    os.rename(join(base_dir, filename), join(target_dir, filename))
                except:
                    print("ERROR: cannot move file {}".format(join(base_dir, filename)))

                    
