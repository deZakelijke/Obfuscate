import facenet_pytorch as fp
import torch
import numpy as np

import csv
import os
from sklearn.neighbors import NearestNeighbors


# UTILS
temporary_location = "target_embeddings"
temporary_csv = "temporary_embeddings.csv"
temporary_path = os.path.join(temporary_location, temporary_csv)


def append_target_embedding(embedding, path="target_embeddings/temporary_embeddings.csv"):
    fields = (['id'] + [str(i) for i in range(128)])
    file_exists = os.path.isfile(path)
    if file_exists:
        with open(path, 'r') as f:
            src_name = int(sum(1 for _ in f.readlines())) # automatically determine id based on number of lines
    else:
        src_name = 1
    with open(path, 'a') as f:
        w = csv.writer(f)
        # write headers only if file doesnt exist
        if not file_exists:
            w.writerow(fields)
        content = tuple([src_name]+ embedding)
        w.writerow(content)


def read_target_embeddings(path="target_embeddings/temporary_embeddings.csv"):
    ids = []
    embeddings = []
    file_exists = os.path.isfile(path)
    if not file_exists:
        return [], []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
        for l in lines[1:]: #skip header line
            ids.append(l[0])
            embeddings.append(l[1:])
    return ids, embeddings


class FacenetClassifier:
    ids = None
    target_embeddings = None
    model = None
    kNN = None
    threshold = None
    csv_location = None

    def __init__(self, location=temporary_location, threshold=0.7):
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.csv_location = location
        os.makedirs(location.split("/")[0], exist_ok=True)

        torch.no_grad()
        self.model = fp.InceptionResnetV1(pretrained='vggface2').eval()
        if torch.cuda.is_available():
            self.model.cuda()

        self.ids, self.target_embeddings = read_target_embeddings(self.csv_location)
        if len(self.ids) > 0:
            self.kNN = self.fit_targets() # we have kNN object after this

    def embed(self, image_array, save=False):
        embedding = self.model(image_array.to(self.device))  # forward pass
        if save:
            # this is BxDim so we must loop
            for i in range(embedding.shape[0]):
                emb = embedding[i].squeeze().tolist()
                append_target_embedding(emb, path=self.csv_location)  # save it to the csv
                self.target_embeddings.append(emb)

        return embedding

    def fit_targets(self):
        embedding_tensor = np.array(self.target_embeddings)
        self.kNN = NearestNeighbors(n_neighbors=1).fit(embedding_tensor)
        return self.kNN

    # expect a BxDimensions tensor
    def match(self, incoming_embeddings):
        # convert to numpy array for the kNN
        incoming_embeddings = incoming_embeddings.cpu().detach().numpy()
        distance, _ = self.kNN.kneighbors(incoming_embeddings)
        # find id of the embeddings with a close enough match
        matching_indices = (np.squeeze(distance) < self.threshold)
        return matching_indices

