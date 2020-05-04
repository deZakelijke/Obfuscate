import cv2
import dlib
import torch
import numpy as np
from mtcnn import MTCNN
from PIL import Image



class Pipeline:
    classifier = None

    def __init__(self, classifier_obj):
        self.classifier = classifier_obj
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor('../preprocessing/shape_predictor_5_face_landmarks.dat')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
            device=self.device)

    def add_to_targets(self, image):
        x_aligned, prob, boxes = self.mtcnn(Image.fromarray(image), return_prob=True, remove_eyes=False, keep_all=False)
        cropped = x_aligned.reshape(1,3,160,160)
        #cropped, _ = self.extract_faces(image)
        #print(cropped.shape)

        if len(cropped) > 1 or len(cropped) == 0:
            print("no face detected")
            return False
        # expect cropped to be BxCxWxH
        self.classifier.embed(cropped, save=True)  # makes an embedding and saves it to csv
        self.classifier.fit_targets()  # make it fit new set of targets

    def pipeline(self, image):
        x_aligned, prob, boxes = self.mtcnn(Image.fromarray(image), return_prob=True, remove_eyes=False, keep_all=True)

        embeddings = self.classifier.embed(x_aligned, save=False)
        matches = self.classifier.match(embeddings)

        if type(matches) == np.bool_:
            matches = [matches]

        places_to_blur = [x[0] for x in zip(boxes, matches) if x[1]]


        blurred_image = self.blur_image(image, places_to_blur)
        return blurred_image

    def extract_faces(self, img):
        dets = self.detector(img, 1)
        if len(dets) == 0:
            return None, None

        face_imgs = None
        rects = []
        for i in range(len(dets)):
            shape = self.sp(img, dets[i])
            rects += [shape.rect]
            face_chip = dlib.get_face_chip(img, shape)
            if face_imgs is None:
                face_imgs = torch.Tensor(face_chip).reshape(1, 3, 150, 150)
            else:
                face_chip = torch.Tensor(face_chip).reshape(1, 3, 150, 150)
                face_imgs = torch.cat((face_imgs, face_chip))

        return face_imgs.to(self.device), np.array(rects)

    # perform some image transformation
    def blur_image(self, image, blur_locations):
        result_image = image
        for loc in blur_locations:
            # loc = loc[0] # TODO: HUHHHH??
            if type(loc) == list:
                loc = loc[0]
            # Get the origin co-ordinates and the length and width till where the face extends

            x = int(loc[0])
            y = int(loc[1])
            w = int(loc[2]) - int(loc[0])
            h = int(loc[3]) - int(loc[1])

            print(x,y,w,h)

            # get the rectangle img around all the faces
            #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 5)
            sub_face = image[y:y + h, x:x + w]
            k_x = w if w % 2 == 1 else w-1
            k_y = h if h % 2 == 1 else h-1
            # apply a gaussian blur on this new rectangle image
            sub_face = cv2.GaussianBlur(sub_face, (k_x, k_y), 0.1 * k_x, 0.1 * k_y)
            # merge this blurry rectangle to our final image
            result_image[y:y + sub_face.shape[0], x:x + sub_face.shape[1]] = sub_face
        return result_image
