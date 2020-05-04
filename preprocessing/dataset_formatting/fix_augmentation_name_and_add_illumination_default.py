import os
from tqdm import tqdm
import argparse


def change_names(loc):
    fs = os.listdir(loc)
    fs = [f for f in fs if f[-3:] == 'png']

    for old_name in fs:

        try:
            name, floatingP, extension = old_name.split('.')
            identifier, sess, pose, ill, exp, p, y, r, illx, intx = name.split("_")

            new_name_l = [identifier, sess, pose, ill, exp, 'p'+p, 'y'+y, 'r'+r, 'il'+illx, 'in'+intx]
            new_name = "_".join(new_name_l) + floatingP + '.' + extension

            # RENAME FILE
            os.rename( loc+'/'+old_name, loc+'/'+new_name)
        except:
            # either the filename is different or we have already processed this earlier
            print("yeet")
            pass


def add_illum_intensity(loc):
    fs = os.listdir(loc)
    fs = [f for f in fs if f[-3:] == 'png']

    for old_name in tqdm(fs):

        try:
            name, extension = old_name.split('.')
            identifier, sess, pose, ill, exp, p, y, r = name.split("_")

            new_name_l = [identifier, sess, pose, ill, exp, p, y, r, 'il0', 'in0']
            new_name = "_".join(new_name_l) + '.' + extension

            # RENAME FILE
            os.rename( loc+'/'+old_name, loc+'/'+new_name)
        except:
            # either the filename is different or we have already processed this earlier
            print("yeet", old_name)
            pass



parser = argparse.ArgumentParser(description='change names of all file in folder')
parser.add_argument('--loc', type=str, default='.', help='location of files')

if __name__ == '__main__':
    args = parser.parse_args()
    add_illum_intensity(args.loc)
