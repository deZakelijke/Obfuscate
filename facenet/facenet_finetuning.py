import argparse
import torch
import torch.optim as optim
import facenet_pytorch as fp
import time
import progressbar
from torch.multiprocessing import Pool, Process, Manager, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

from .dataloader_finetuning import *
from .make_facenet_embeddings import map_images_to_embedding

FINETUNED_FILENAME="finetuned_2.pt"

class FacenetFinetuning:
    model = None
    dl = None
    dataset = None
    load_images = None
    source_dir = None
    augmentation_dir = None
    num_workers = None
    batch_size = None

    def __init__(self, args, load_images=True):
        self.source_dir = args.source_dir
        self.augmentation_dir = args.augmentation_dir
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.dl, self.dataset = get_easy_dataloader(self.source_dir, self.augmentation_dir,
                                                    load_images=load_images, batch_size=self.batch_size,
                                                    num_workers=self.num_workers)

        self.model = self.init_model()
        self.load_images = load_images

    def init_model(self):
        model = fp.InceptionResnetV1(pretrained='vggface2').to(device)
        # load pre trained weights
        if os.path.exists(FINETUNED_FILENAME):
            model.load_state_dict(torch.load(FINETUNED_FILENAME, map_location=device))
        model.train()
        model = self.freeze_layers(model)
        return model

    def freeze_layers(self, model):
        for param in model.parameters():
            param.requires_grad = False

        # Freeze everything except for the last layers
        model.last_linear.requires_grad = True
        model.last_bn.requires_grad = True
        return model

    def get_neighbour(self, sample, embeddings, naming_list, mode="far"):
        c = embeddings.cuda() - sample.cuda()
#        c = embeddings - sample
        distances = torch.norm(c, dim=1)
        if mode == "far":
            match = distances.argmax()
            return naming_list[match]
        else:
            match = distances.argmin()
            return naming_list[match]

    def get_embedding_for_filenames(self, filenames, embeddings, naming_dict):
        embeddings_ = []
        indices = []

        for p in filenames:
            e_idx = naming_dict[p]
            indices.append(e_idx)
            e = embeddings[e_idx]
            embeddings_.append(e)
        return torch.stack(embeddings_), indices

    def determine_triplets(self, q, naming_list, full_naming_list, embeddings,  naming_dict,  dataset):
        triplets = []
        for name in naming_list:
            sample_embedding = embeddings[naming_dict[name]]

            # get filenames that are of the same class
            positives = dataset.get_positive(filename=name)
            if len(positives) == 0:
                print("no positives found")
            positive_embeddings, pos_indices = self.get_embedding_for_filenames(positives, embeddings, naming_dict)
            pos_names = [full_naming_list[p] for p in pos_indices]
            positive = self.get_neighbour(sample_embedding, positive_embeddings, pos_names, mode="far") #returns filename

            # get filenames that are of different classes
            negatives = dataset.get_negative(filename=name)
            if len(negatives) == 0:
                print("no negatives found")

            negative_embeddings, neg_indices = self.get_embedding_for_filenames(negatives, embeddings, naming_dict)
            neg_names = [full_naming_list[p] for p in neg_indices]

            negative = self.get_neighbour(sample_embedding, negative_embeddings, neg_names, mode="close") #returns filename

            triplets.append((name, positive, negative))
            q.put(1)
        q.put(0)
        return triplets

    def start_multiprocessing(self, embeddings, naming_list, naming_dict,  dataset):
        n_threads = 4
        chunked = chunkIt(naming_list, n_threads)
        # multiprocess gridsearch and have a seperate thread for the progress bar.
        pool1 = Pool(processes=n_threads)
        m = Manager()
        q = m.Queue()
        p = Process(target=progressBar, args=(len(naming_list), q,))
        p.start()

        results = pool1.starmap(self.determine_triplets,
                                zip(n_threads * [q],
                                    chunked,
                                    n_threads * [naming_list],
                                    n_threads * [embeddings],
                                    n_threads * [naming_dict],
                                    n_threads * [dataset]))
        final_results = []
        for r in results:
            final_results += r

        p.join()
        pool1.close()
        return final_results

    def get_hard_dataloader(self, full_dataloader, dataset):
        embeddings, naming_list = map_images_to_embedding(self.model, full_dataloader)
        naming_dict = {k : v for v, k in enumerate(naming_list)}

        # triplets = self.determine_triplets(embeddings, naming_dict, naming_list, dataset)
        triplets = self.start_multiprocessing(embeddings, naming_list, naming_dict, dataset)
        # triples is a list of tuples (sample, pos, neg)
#        print("got triplets", len(triplets))
        dataloader = get_train_dataloader(triplets, load_images=self.load_images, batch_size=self.batch_size)
        return dataloader


    def triplet_loss(self, anchor, positive, negative, alpha):
#        print(anchor.shape, positive.shape, negative.shape)
        if len(positive) < len(anchor) or len(positive) < len(negative):
            print("len positive too short?!?!")
            anchor = anchor[:len(positive)]
            negative = negative[:len(positive)]
        pos_dist = torch.norm(anchor-positive, dim=1)
        neg_dist = torch.norm(anchor-negative, dim=1)
        loss = (pos_dist - neg_dist + alpha).relu()
#        print(loss.shape)
        return torch.mean(loss)


    def train(self, epochs=1, alpha=0.2, learning_rate=1e-5, print_every=10):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    #    optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        criterion = self.triplet_loss
        for i in range(epochs):
            t = time.time()
            print("getting dataloader")
            self.dl, self.dataset = get_easy_dataloader(self.source_dir, self.augmentation_dir,
                                                        load_images=self.load_images, batch_size=self.batch_size,
                                                        num_workers=self.num_workers)

            hard_dataloader = self.get_hard_dataloader(self.dl, self.dataset)
            print("got dataloader")
            for index, (anchor, pos, neg) in enumerate(hard_dataloader):
#                print("train loop iteration", index)
                input = torch.cat([anchor["image"], pos["image"], neg["image"]], dim=0).to(device)
#                print("input shape", input.shape)
                input.requires_grad=True
                embeds = self.model(input)
                ax = embeds[:self.batch_size]
                px = embeds[self.batch_size:self.batch_size*2]
                nx = embeds[-self.batch_size:]
#                print(ax.shape, px.shape, nx.shape, len(ax)+len(px)+len(nx), batch_size*3)
                loss = criterion(ax, px, nx, alpha)
                if index % print_every == 0:
                    print("Loss:", loss.item())
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
            print("Finished epoch:", i, "took", time.time()-t)
            torch.save(self.model.state_dict(), FINETUNED_FILENAME)


def progressBar(max_value, queue):
    bar = progressbar.ProgressBar(max_value=max_value).start()
    finished = 0
    while finished < 4:
        if queue.get() == 1:
            bar += 1
        else:
            finished += 1

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default="/home/douwe/Documents/facenet_preprocessed",
                        help='directory with crowd images')
    parser.add_argument('--augmentation_dir', type=str, default=None,
                        help='directory with crowd images')
    parser.add_argument('--number_of_targets', type=int, default=100,
                        help='this will determine the number of unique ids')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-05, help="learning rate")
    parser.add_argument('--num_workers', type=int, default=4, help="number of workers for the dataloader")
    parser.add_argument('--epochs', type=int, default=1, help="number of epochs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args, unparsed = parser.parse_known_args()

    tuner = FacenetFinetuning(args)
    tuner.train(learning_rate=args.learning_rate, epochs=args.epochs)
