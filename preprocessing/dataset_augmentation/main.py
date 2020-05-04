import time
import argparse

from utils import prepare_batch, unzip_and_move, get_files, copy_originals, number_of_files, pretty_print
from universum_api import post_batch, get_status, get_processed

def main():
    # get file iterator
    source_loc = "dataset" if ARGS.source == "" else ARGS.source
    target_loc = "dataset_processed"
    bundle_loc = "batch.zip"

    processed_loc = "batch_processed.zip"

    batch_size = 25
    poses = [
        {
            "yaw": -8,
            "pitch": -8,
            "roll": 0
        },
        {
            "yaw": -1,
            "pitch": 8,
            "roll": 0
        },
        {
            "yaw": 5,
            "pitch": 5,
            "roll": 0
        }
    ]

    total_files = number_of_files(source_loc)
    counter = 0
    current_progress = 0
    iterator = get_files(source_loc, batch_size=batch_size)

    for batch_filenames in iterator:
        current_progress += len(batch_filenames)
        pretty_print(counter, "processing: {}/{}".format(current_progress, total_files))
        prepare_batch(batch_filenames, bundle_loc)
        pretty_print(counter, "posting batch request")
        process_id = post_batch(poses=poses, bundle_loc=bundle_loc)
        pretty_print(counter, "request id: {}".format(process_id))

        # do some waiting until it is finished
        finished = False
        while not finished:
            finished = get_processed(process_id, processed_loc)  # downloads the zip
            time.sleep(10)

        # pretty_print(counter, "{} is available".format(process_id))
        # get_processed(process_id, processed_loc)  # downloads the zip
        names = [name.split('/')[1].split('.')[0] for name in batch_filenames]
        # extract the zip, go into the 3duniversum structure and move the files to the target location
        unzip_and_move(processed_loc, target_loc, names=names)
        # Copy the originals
        copy_originals(batch_filenames, target_loc)
        pretty_print(counter, "batch finished")

        counter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default="", type=str,
                        help='location of the faces to augment')
    ARGS = parser.parse_args()
    main()
