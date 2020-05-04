import torch
import numpy as np
import pickle

from sklearn.neighbors import NearestNeighbors
from prediction import Prediction, write_to_csv


def get_labels_from_naming_data(naming_data):
    labels = np.array([int(x.split('/')[-1].split('_')[0]) for x in naming_data])
    return np.array(labels)


def compute_accuracy(predicted_classes, labels, threshold, distance, source_naming_list, \
                     target_indices, match_index, target_naming_list):
    """ Compute the accuracy for classifying the source items as elements of targets.
    """
    predictions = []

    predicted_classes[np.squeeze(distance) > threshold] = -1
    correct = np.equal(predicted_classes, labels)
    try:
        accuracy = sum(correct) / len(correct)
    except ZeroDivisionError:
        accuracy = 0.0
    correct_targets = predicted_classes[target_indices] == labels[target_indices]
    try:
        targets_accuracy = sum(correct_targets) / len(correct_targets)
    except ZeroDivisionError:
        targets_accuracy = 0.0

    for j in range(len(source_naming_list)):
        tar_name = target_naming_list[match_index[j][0]].split('/')[-1] if predicted_classes[j] != -1 else '-1_'
        dist = distance[j] if predicted_classes[j] != -1 else -1
        if dist is None:
            raise ValueError("Invalid distance")
        new_prediction = Prediction(source_naming_list[j].split('/')[-1], dist,
                                    0.0, correct[j], target_indices[j], tar_name)
        predictions.append(new_prediction)
    return accuracy, targets_accuracy, predictions


class FacenetBenchmark:
    target_dlr = None
    base_threshold = None

    def __init__(self, source_dlr, target_dlr,
                 source_filename="results/FaceNet_data",
                 experiment_name="facenet",
                 base_threshold=0.4,
                 threshold_range=1,
                 experiment_extension="base"):
        self.base_threshold = base_threshold
        self.threshold_range = threshold_range
        self.source_dlr = source_dlr
        self.target_dlr = target_dlr
        self.source_filename = source_filename
        self.experiment_name = experiment_name
        self.experiment_extension = experiment_extension

    def get_filtered_embeddings(self, source_file, dataloader):
        source_embeddings = torch.load(f"{source_file}_embeddings.pt").detach().numpy()
        # naming dict: keys are filenames, values are indices
        # naming list: list of filenames
        # indices of naming dict are indices of embeddings
        source_naming_list, source_naming_dict = pickle.load(open(f"{source_file}_embedding_names.pt", "rb"))
        # first obtain the labels
        source_labels = get_labels_from_naming_data(source_naming_list)
        source_naming_list = np.array(source_naming_list)

        embedding_indices = []

        for f in dataloader:
            for filename in f["filename"]:
                # get embedding for this filename
                if filename not in source_naming_dict: continue
                embedding_index = source_naming_dict[filename]
                embedding_indices.append(embedding_index)

        embeddings = source_embeddings[embedding_indices]
        labels = source_labels[embedding_indices]
        naming_list = source_naming_list[embedding_indices]

        return embeddings, labels, naming_list

    def classify(self, target_classes):
        print("loading embeddings")
        source_embeddings, source_labels, source_naming_list = self.get_filtered_embeddings(self.source_filename, self.source_dlr)
        target_embeddings, target_labels, target_naming_list = self.get_filtered_embeddings(self.source_filename, self.target_dlr)
        print("fitting kNN")
        kNN = NearestNeighbors(n_neighbors=1).fit(target_embeddings)
        print("fitting kNN: done")

        distance, match_index = kNN.kneighbors(source_embeddings)

        print("found kNN matches")
        distance = np.array(distance)
        target_indices = np.in1d(source_labels, np.array(list(target_classes)))
        source_labels[~target_indices] = -1
        predicted_classes = np.squeeze(target_labels[match_index])
        print("calculating accuracies")
        best_accuracy = 0
        best_targets_accuracy = 0
        for i in range(self.threshold_range):
            threshold = self.base_threshold + i * 0.02
            accuracy, targets_accuracy, predictions = compute_accuracy(predicted_classes.copy(), source_labels,
                                                     threshold, distance, source_naming_list,
                                                     target_indices, match_index, target_naming_list)
            if accuracy > best_accuracy:
                best_targets_accuracy = targets_accuracy
                print(f"Accuracy: {accuracy:.3f}, Targets accuracy: {targets_accuracy:.3f}, Threshold: {threshold:.2f}")
                best_accuracy = accuracy
                best_threshold = threshold
                best_predictions = predictions
        write_to_csv(best_predictions, model_name=f"FaceNet_threshold_{best_threshold:.2f}_{self.experiment_extension}", location=f"results/{self.experiment_name}")
        print(f"Best accuracy: {best_accuracy:.3f}, Best target accuracy: {best_targets_accuracy:.3f} \
                Threshold: {best_threshold:.2f}")
