import numpy as np
import os
import progressbar
from multiprocessing import Pool, Manager, Process
import dlib
import openface
import cv2

np.set_printoptions(precision=2)

class PreprocessDlibResnet:
    def __init__(self, source_folder, destination_folder):
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
        self.n_threads=1
        self.source_folder = source_folder
        self.destination_folder = destination_folder

    def process_img(self, imgPaths, progress_q):
        """
        :param source: source (RGB) image file to be preprocessed
        :param target: target (RGB) preprocessed image file
        """
        failed_names = []
        for imgPath in imgPaths:

            img = dlib.load_rgb_image(self.source_folder + '/' + imgPath)
            dets = self.detector(img, 2)  # Note: 2 = num jitters, higher is better at finding faces but more expensive
            try:
                shape = self.sp(img, dets[0])  # Note: assume only 1 face present, which is dets[0]
                face_chip = dlib.get_face_chip(img, shape)
                dlib.save_image(face_chip, self.destination_folder + "/" + imgPath)
            except IndexError:
                failed_names.append(imgPath)
            progress_q.put(1)
        progress_q.put(0)
        return failed_names

    def batch_process(self):
        all_file_names = os.listdir(self.source_folder)
        chunked = self.chunkIt(all_file_names, self.n_threads)

        # multiprocess gridsearch and have a seperate thread for the progress bar.
        pool1 = Pool(processes=self.n_threads)
        m = Manager()
        q = m.Queue()
        p = Process(target=self.progressBar, args=(len(all_file_names), q,))
        p.start()

        results = pool1.starmap(self.process_img, zip(chunked, self.n_threads * [q]))
        with open("failed_img_names.txt", "w") as f:
            for result in results:
                for filename in result:
                    f.write(filename + "\n")

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


class AlignMultipleFacesOpenface:
    def __init__(self, dlibFacePredictor=None, imgDim=96):
        # Class specific imports

        self.fileDir = os.path.dirname(os.path.realpath(__file__))
        self.modelDir = os.path.join(self.fileDir, '..', 'models')
        self.dlibModelDir = os.path.join(self.modelDir, 'dlib')
        self.openfaceModelDir = os.path.join(self.modelDir, 'openface')
        if dlibFacePredictor == None:
            self.dlibFacePredictor = os.path.join(self.dlibModelDir, "shape_predictor_68_face_landmarks.dat")
        else:
            self.dlibFacePredictor = dlibFacePredictor
        self.align = openface.AlignDlib(self.dlibFacePredictor)
        self.basepath = "/home/douwe/Documents/crowd_mugshots/"
        self.croppedpath = "/home/douwe/Documents/openface_cropped_crowd_mugshots/"
        self.n_threads = 15
        self.imgDim = imgDim

    def getRep(self, imgPaths, progress_q):
        failed_names = []
        for imgPath in imgPaths:
            bgrImg = cv2.imread(self.basepath + "/" + imgPath)
            if bgrImg is None:
                raise Exception("Unable to load image: {}".format(imgPath))

            rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

            bb1 = self.align.getLargestFaceBoundingBox(rgbImg)
            bbs = [bb1]
            if len(bbs) == 0:
                failed_names += [imgPath]
                # raise Exception("Unable to find a face: {}".format(imgPath))
                continue

            for bb in bbs:
                alignedFace = self.align.align(
                    self.imgDim,
                    rgbImg,
                    bb,
                    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                if alignedFace is None:
                    failed_names += [imgPath]
                    continue
                    #raise Exception("Unable to align image: {}".format(imgPath))

                cv2.imwrite(self.croppedpath + "/" + imgPath, alignedFace)
            progress_q.put(1)
        progress_q.put(0)
        return failed_names

    def chunkIt(self, seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out

    def progressBar(self, max_value, queue):
        bar = progressbar.ProgressBar(max_value=max_value).start()
        finished = 0
        while finished < self.n_threads:
            if queue.get() == 1:
                bar += 1
            else:
                finished += 1

    def run(self):
        all_file_names = os.listdir(self.basepath)
        chunked = self.chunkIt(all_file_names, self.n_threads)

        # multiprocess gridsearch and have a seperate thread for the progress bar.
        pool1 = Pool(processes = self.n_threads)
        m = Manager()
        q = m.Queue()
        p = Process(target=self.progressBar, args=(len(all_file_names), q,))
        p.start()

        results = pool1.starmap(self.getRep, zip(chunked, self.n_threads * [q]))
        with open("failed_img_names.txt", "w") as f:
            for result in results:
                for filename in result:
                    f.write(filename + "\n")

        p.join()
        pool1.close()


if __name__ == '__main__':
    # Example code of how to preprocess a file named tom.jpeg for the dlib model
    prep_dlib = PreprocessDlibResnet(source_folder='sample_images', destination_folder='dlibresnet_preprocessing')
    prep_dlib.batch_process()
