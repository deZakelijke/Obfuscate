from multiprocessing import Pool, Process, Manager
import numpy as np
import progressbar
import argparse
import pickle
import dlib
import os

from prediction import Prediction
from dataloader import get_dataloader


class DlibResnet:
    def __init__(self, model_path, n_threads=1):
        self.model_path = model_path
        self.face_embeddings = []
        self.embedding_ids = []
        self.unique_ids = set()
        self.n_threads = n_threads


    def fit(self, data_loc, save_embedding, embedding_dir):
        face_rec = dlib.face_recognition_model_v1(self.model_path)

        file_names = os.listdir(data_loc)
        for file_name in file_names:
            img = dlib.load_rgb_image(data_loc + "/" + file_name)
            embedding = face_rec.compute_face_descriptor(img)
            self.face_embeddings.append(embedding)
            self.embedding_ids.append(file_name)

        if save_embedding:
            pickle.dump((self.face_embeddings, self.embedding_ids), open(embedding_dir, 'wb'))


    def load_embeddings(self, data_loc):
        """Any embedding pickle should look like
            [(EMBEDDINGS), (EMBEDDING_FILE_NAMES)]
            From these, we can extract a list of unique names for prediction
        """
        self.face_embeddings, self.embedding_ids = pickle.load(open(data_loc, 'rb'))
        self.unique_ids = set(self.embedding_ids)


    def predict(self, file_names, q, fault_tolerance=0.6):
        assert self.face_embeddings is not None, "load embeddings first!"
        face_rec = dlib.face_recognition_model_v1(self.model_path)
        results = []
        embeddings = []
        embedding_ids = []

        for source_name in file_names:
            # Save the ID/class of the current file

            img = dlib.load_rgb_image(self.data_loc + "/" + source_name)
            embedding = face_rec.compute_face_descriptor(img)
            embeddings.append(embedding)
            embedding_ids.append(source_name)

            # Calculate distances and threshold to tolerance level
            distances = np.linalg.norm(np.stack(self.face_embeddings)-embedding, axis=1)
            pred_results = distances <= fault_tolerance
            matches = [(i, d) for (i, j, d) in zip(self.embedding_ids, pred_results, distances) if j]

            if len(matches) == 0:
                target_name = '-1_'
                correct = True if source_name in self.unique_ids else False
                distance = -1
            else:
                target_name, distance = min(matches, key=lambda x: x[1])
                source_ID = int(source_name.split('_')[0])
                target_ID = int(target_name.split('_')[0])
                correct = (source_ID == target_ID)

            results.append(Prediction(
                source_name=source_name,
                distance=distance,
                correct=correct,
                confidence=-1,
                target_name=target_name
            ))
            q.put(1)
        q.put(0)

        return results, np.array(embeddings), embedding_ids


    def batch_process(self, args):
        self.data_loc = args.prediction_folder
        all_file_names = os.listdir(self.data_loc)
        chunked = self.chunkIt(all_file_names, self.n_threads)

        # multiprocess gridsearch and have a seperate thread for the progress bar.
        pool1 = Pool(processes=self.n_threads)
        m = Manager()
        q = m.Queue()
        p = Process(target=self.progressBar, args=(len(all_file_names), q,))
        p.start()
        results = pool1.starmap(self.predict, zip(chunked, self.n_threads * [q]))

        embeddings = np.concatenate([i for (_, i, _) in results])
        embedding_ids = [i for (_, _, i) in results]
        predictions = [i for (i, _, _) in results]

        pickle.dump((embeddings, embedding_ids), open(args.prediction_embeddings_loc, 'wb'))
        pickle.dump(predictions, open(args.prediction_loc, 'wb'))

        p.join()
        pool1.close()


    def progressBar(self, max_value, queue):
        bar = progressbar.ProgressBar(max_value=max_value).start()
        finished = 0
        while finished < self.n_threads:
            if queue.get() == 1:
                bar += 1
            else:
                finished += 1


    def chunkIt(self, seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out


parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dir', type=str, default="dlibResnet/face_embeddings_target_mugshot_dlibResnet.pkl",
                    help='location')
parser.add_argument('--prediction_folder', type=str, default='results/mugshots/')
parser.add_argument('--prediction_loc', type=str, default='dlibResnet/predictions.pkl')
parser.add_argument('--prediction_embeddings_loc', type=str, default='dlibResnet/face_embeddings_predictions')
parser.add_argument('--n_threads', type=int, default=2)
parser.add_argument('--model_dir', type=str, default='dlibResnet/dlib_face_recognition_resnet_model_v1.dat')
parser.add_argument('--threshold', type=float, default=0.6, help='Threshold for euclidean distance prediction')

if __name__ == '__main__':
    args = parser.parse_args()
    # dlib_model = DlibResnet(model_path=args.model_dir, n_threads=args.n_threads)
    #
    # # If you need to still create embeddings for the 'targets':
    # #dlib_model.fit(data_loc=args.prediction_folder, save_embedding=True, embedding_dir=args.embedding_dir)
    #
    # dlib_model.load_embeddings(data_loc=args.embedding_dir)
    # dlib_model.batch_process(args)

    source_dl, target_dl, target_class = get_dataloader(args.prediction_folder, number_of_targets=30)
    for a in target_dl:
        print(a)
