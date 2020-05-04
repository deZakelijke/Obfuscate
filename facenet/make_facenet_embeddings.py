import os
import facenet_pytorch as fp
import torch
import numpy as np
import pickle
import argparse
import time
import progressbar

from PIL import Image
from torch.utils.data import DataLoader
from torch.nn.functional import pdist, pairwise_distance
from torchvision import transforms
from dataloader import get_dataloader
from sklearn.neighbors import NearestNeighbors
from prediction import Prediction, write_to_csv

def map_images_to_embedding(model, dataloader):
    """ Create embeddings for all images in the dataloader.
    """
    bar = progressbar.ProgressBar(max_value=len(dataloader))
    bar.start()
    naming_list = []
    total_embeddings = []
    embedding_list = []

    for index, batch in enumerate(dataloader):
        bar.update(index + 1)
        image = batch['image']
        naming_data = batch['filename']
        if torch.cuda.is_available():
            image = image.cuda()
        output = model(image)
        embedding_list.append(output)
        naming_list += naming_data

        if (index + 1) % len(batch) == 0:
            total_embeddings.append(torch.cat(embedding_list).view(-1, 512).cpu())
            embedding_list = []
           
    try:
        total_embeddings.append(torch.cat(embedding_list).view(-1, 512).cpu())
    except RuntimeError:
        pass 

    
    bar.finish()
    return torch.cat(total_embeddings).view(-1, 512), naming_list

def get_embeddings(source_dlr, target_dlr, args):
    with torch.no_grad():
        resnet = fp.InceptionResnetV1(pretrained='vggface2').eval()
        if torch.cuda.is_available():
            resnet.cuda()

        if args.pretrained != "" and os.path.exists(args.pretrained):
            resnet.load_state_dict(torch.load(args.pretrained))
        source_embeddings, source_names = map_images_to_embedding(resnet, source_dlr)
        #target_embeddings, target_names = map_images_to_embedding(resnet, target_dlr, args, target=True)
        torch.save(source_embeddings, f"{args.source_filename}_embeddings.pt")

        naming_list = source_names #+ target_names
        naming_dict = {k : v for v, k in enumerate(naming_list)}
        pickle.dump((naming_list, naming_dict), open(f"{args.source_filename}_embedding_names.pt", "wb"))

    print("New embeddings created")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default="/home/douwe/Documents/facenet_preprocessed/",
                        help='directory with crowd images')
    parser.add_argument('--source_filename', type=str, default='results/FaceNet_data_source',
                        help='prefix for the storing of the embeddings')
    parser.add_argument('--nr_targets', type=int, default=100,
                        help='number of target images') # is prob deprecated
    parser.add_argument('--nr_source_images', type=int, default=10,
                        help='number of source images') # is prob deprecated
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--pretrained', type=str, default="",
                        help='location of the finetuned/pretrained weights')
    args, unparsed = parser.parse_known_args()


    source_dlr, target_dlr, target_classes = get_dataloader(args.source_dir, load_images=True,
                                                            batch_size=args.batch_size, 
                                                            shuffle=False, seed=int(time.time()))

    get_embeddings(source_dlr, target_dlr, args)
