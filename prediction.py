import csv
import os
from collections import namedtuple

class Prediction:
    """
    Sample code, given the file we want to match as 'file', and the highest prediction as 'prediction':
    It is important to either use the named arguments or to follow the order in the class instead.

    You need to keep track of the file names for the targets as well, in a dictionary.
    If you do not have a match, i.e. it does not exceed the distance threshold,
    you need to either provide target_name="-1_" or not pass this argument.

    You also need to determine whether the prediction was correct yourself, because we can no longer infer this
    after the target_name has been overwritten.

    To do this, test if the prediction exceeds the threshold, if it does: perform a look up and see if they match.
    If they match, the prediction is correct.

    If the prediction does not exceed the threshold, and the source label (e.g. 004) you are matching is not in
    your dictionary of targets, it means that it correctly did not assign it a label.
    This means the prediction is correct.

    If the prediction does not exceed the threshold but the source label is in the target dictionary,
    it should have actually linked it to a target, so this prediction is incorrect.

    Create an object like so: `id_pose_illum_expression_px_yx_rx_ilx_lx.png`
    The `id_session_illum_expression` are the names of the files we provide.
    The `px_yx_rx` correspond to the viewpoint changes pitch, yaw, roll, as your API outputs.
    The `ilx` corresponds to illumination <number> where 0=no change, 1=Left side, 2=Right, and 3=Front.
    The `lx` corresponds to intensity

    prediction = Prediction(source_name=source, target_name=target, distance=d, confidence=None, correct=c)
    results.append(prediction)

    provide the filename according to the convention, we extract the value
    in the postprocessing script.
    """

    def __init__(self, source_name, distance, confidence, correct, is_target, target_name='-1_'):
        self.source_name = str(source_name)
        self.target_name = str(target_name)
        self.correct = correct
        self.is_target = is_target

        # make sure to only cast when argument filled
        self.distance = None if not distance else float(distance)
        self.confidence = None if not confidence else float(confidence)

        self.true_class = int(source_name.split('_')[0])
        self.predicted_class = int(target_name.split('_')[0])


# call this with the model name to make sure we don't overwrite results
def write_to_csv(list_of_predictions, model_name="default", location="results"):
    fields = ("source_name", "target_name", "distance", "confidence", "predicted_class", "true_class", "correct", "is_target")
    output_name = "{}_{}.csv".format("predictions", model_name)
    os.makedirs(location, exist_ok=True)
    path = os.path.join(location, output_name)
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerow(fields)  # field header
        rows = []
        for p in list_of_predictions:
            if p.source_name is None:
                print("missing source_name in prediction")
            if p.target_name is None:
                print("missing target_name in prediction")
            if p.distance is None and p.confidence is None:
                print("distance and confidence can not both be None")
            if p.predicted_class is None:
                print("missing predicted_class")
            if p.true_class is None:
                print("missing true_class")
            if p.correct is None:
                print("missing correct: correctness of prediction")
            if p.is_target is None:
                print("missing is_target: whether this source has a target or not")
            row = (p.source_name,
                   p.target_name,
                   p.distance,
                   p.confidence,
                   p.predicted_class,
                   p.true_class,
                   p.correct,
                   p.is_target)
            rows.append(row)

        w.writerows(rows)  # dirty extraction and you know it
    print("Wrote {} predictions to {}".format(len(list_of_predictions), path))
