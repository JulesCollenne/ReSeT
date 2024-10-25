import random

import numpy as np
import pandas as pd


class DataLoaderISIC:
    def __init__(self, df_name, gt_name, batch_size, n_vect=100):
        self.batch_size = batch_size
        self.n_vect = n_vect

        self.gt = pd.read_csv(gt_name)
        self.train_features = pd.read_csv(df_name)

        matching_patient_ids = self.gt[self.gt["image_name"].isin(self.train_features["image_name"])]["patient_id"]
        self.patients = matching_patient_ids.unique()
        np.random.shuffle(self.patients)
        self.n_features = sum(['feature' in col for col in self.train_features.columns])

    def data(self):
        start = 0
        end = self.batch_size
        while True:
            patients_names = self.patients[start:end]
            patients_imgs = [self.gt[self.gt["patient_id"] == pid]["image_name"].values for pid in patients_names]
            current_features = []
            current_labels = []
            for pnum, current_patient_imgs in enumerate(patients_imgs):
                random.shuffle(current_patient_imgs)
                current_patient_features = []
                current_patient_labels = []
                for i in range(self.n_vect):
                    if i < len(current_patient_imgs):
                        current_patient_features.append(
                            self.train_features[self.train_features["image_name"] == current_patient_imgs[i]].filter(
                                like="feature").values[0])
                        current_patient_labels.append(
                            self.gt[self.gt["image_name"] == current_patient_imgs[i]]["target"].values[0])
                    else:
                        current_patient_features.append(np.zeros(self.n_features))
                        current_patient_labels.append(0)
                current_features.append(current_patient_features)
                current_labels.append(current_patient_labels)
            current_features = np.asarray(current_features)
            current_labels = np.asarray(current_labels)
            yield current_features, current_labels
            if end >= len(self.patients):
                break
            end += self.batch_size
            start += self.batch_size

    def test_data(self):
        # This function is used to make sure that we process all images of the patients
        start = 0
        end = self.batch_size
        while True:
            patients_names = self.patients[start:end]
            patients_imgs = [self.gt[self.gt["patient_id"] == pid]["image_name"].values for pid in patients_names]
            current_features = []
            current_labels = []
            for pnum, current_patient_imgs in enumerate(patients_imgs):
                random.shuffle(current_patient_imgs)
                n_pass = 0
                while n_pass < len(current_patient_imgs):
                    current_patient_features = []
                    current_patient_labels = []
                    for i in range(self.n_vect):
                        if i < len(current_patient_imgs):
                            current_patient_features.append(
                                self.train_features[self.train_features["image_name"] == current_patient_imgs[i]].filter(
                                    like="feature").values[0])
                            current_patient_labels.append(
                                self.gt[self.gt["image_name"] == current_patient_imgs[i]]["target"].values[0])
                        else:
                            current_patient_features.append(np.zeros(self.n_features))
                            current_patient_labels.append(0)
                    current_features.append(current_patient_features)
                    current_labels.append(current_patient_labels)
                    n_pass += self.n_vect
            current_features = np.asarray(current_features)
            current_labels = np.asarray(current_labels)
            yield current_features, current_labels
            if end >= len(self.patients):
                break
            end += self.batch_size
            start += self.batch_size
