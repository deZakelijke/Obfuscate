import zipfile
import os
from datetime import datetime
from shutil import copyfile, rmtree

def pretty_print(count, text):
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = "{}\t|\t{}\t|\t{}".format(time, count, text)
    print(formatted)

def number_of_files(source_loc):
    files = os.listdir(source_loc)
    return len(files)

def get_files(source_loc, batch_size=25):
    files = os.listdir(source_loc)
    files = sorted(files)
    for i in range(0, len(files), batch_size):
        # Create an index range for l of n items:
        yield [os.path.join(source_loc, f) for f in files[i:i + batch_size]]


def prepare_batch(source_loc, target_loc):
    zipf = zipfile.ZipFile(target_loc, 'w')
    for f in source_loc:
        zipf.write(f, os.path.basename(f))

    zipf.close()


def copy_originals(source_loc, target_loc):
    default_values = "_p0_y0_r0.png"
    for filename in source_loc:
        # remove source folder, adjust extension to contain the default params
        filename_split = filename.split('/')[1].split('.')[0] + default_values
        copyfile(filename, os.path.join(target_loc, filename_split))


def unzip_and_move(source_loc, target_loc, names):
    os.makedirs(target_loc, exist_ok=True)
    tmp_loc = "tmp_processed"
    default = "default.png"

    zip_ref = zipfile.ZipFile(source_loc, 'r')
    zip_ref.extractall(tmp_loc)
    zip_ref.close()

    # extract images from target loc
    folders = sorted(os.listdir(tmp_loc))
    folders = list(filter(lambda x: ".DS_Store" not in x, folders))
    for index, folder in enumerate(folders):
        files = os.listdir(os.path.join(tmp_loc, folder))
        longer_than = len(default)

        # extract only the viewpoint augmentations
        filtered = list(filter(lambda x: (".png" in x
                                          and "Enzo" not in x
                                          and not ".DS_Store" in x)
                                         and len(x) > longer_than, files))

        # move the augmented images
        for filename in filtered:
            filename_split = filename.split('default')[1]  # skip the default from the name
            os.rename(os.path.join(tmp_loc, folder, filename),
                      os.path.join(target_loc, "{}{}".format(names[index], filename_split)))

        # copy the original file with name the default params
        # default_values = "_p0_y0_r0.png"
        # original_copy = names[index].split('.')[0] + default_values
        # copyfile(os.path.join(tmp_loc, folder, default),
        #          os.path.join(target_loc, original_copy))

    rmtree(tmp_loc) # clean remaining files?
