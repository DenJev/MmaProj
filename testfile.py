import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import six
from six.moves import cPickle as pickle
import model
# noinspection PyPep8
import string

from sklearn.neural_network import MLPClassifier

#full_data_frame = pd.read_csv('full_data_frame.csv')
with open('my_dumped_classifier.pkl', 'rb') as fid:
    gnb_loaded = pickle.load(fid)

full_data_frame = pd.DataFrame()
fighter_loop = list(csv.reader(open('middleweight_fighters.csv', 'rt', encoding = 'GBK')))
for i in range(1, len(fighter_loop)):
    object_choice = model.fight_manipulation(fighter_loop[i][1])
    object_choice.filtered_fights('DEC')
    object_choice.filtered_fights_three_round()
    object_choice.winner()

    object_choice.attribute_difference("B_Age")
    object_choice.attribute_difference('B_Height')
    object_choice.attribute_difference('BPrev')
    object_choice.attribute_difference('BStreak')
    object_choice.attribute_difference('B__Round1_Strikes_Body Total Strikes_Attempts')
    object_choice.attribute_difference('B__Round1_Strikes_Body Total Strikes_Landed')
    object_choice.attribute_difference('B__Round1_Strikes_Ground Total Strikes_Attempts')
    object_choice.attribute_difference('B__Round1_Strikes_Ground Total Strikes_Landed')
    object_choice.attribute_difference('B__Round1_TIP_Ground Control Time')
    object_choice.attribute_difference('B__Round1_Grappling_Takedowns_Attempts')
    object_choice.attribute_difference('B__Round1_Grappling_Takedowns_Landed')
    object_choice.add_fighter_type()
    # print object_choice.chosen_attributes
    # print object_choice.chosen_attributes['Striker']
    object_choice.attribute_difference('B__Round2_Strikes_Body Total Strikes_Attempts')
    object_choice.attribute_difference('B__Round2_Strikes_Body Total Strikes_Landed')
    object_choice.attribute_difference('B__Round2_Strikes_Ground Total Strikes_Attempts')
    object_choice.attribute_difference('B__Round2_Strikes_Ground Total Strikes_Landed')
    object_choice.attribute_difference('B__Round2_TIP_Ground Control Time')
    object_choice.attribute_difference('B__Round2_Grappling_Takedowns_Attempts')
    object_choice.attribute_difference('B__Round2_Grappling_Takedowns_Landed')
    object_choice.attribute_difference('B__Round3_Strikes_Body Total Strikes_Attempts')
    object_choice.attribute_difference('B__Round3_Strikes_Body Total Strikes_Landed')
    object_choice.attribute_difference('B__Round3_Strikes_Ground Total Strikes_Attempts')
    object_choice.attribute_difference('B__Round3_Strikes_Ground Total Strikes_Landed')
    object_choice.attribute_difference('B__Round3_TIP_Ground Control Time')
    object_choice.attribute_difference('B__Round3_Grappling_Takedowns_Attempts')
    object_choice.attribute_difference('B__Round3_Grappling_Takedowns_Landed')

    full_data_frame = full_data_frame.append(object_choice.chosen_attributes)
    full_data_frame = full_data_frame.append(object_choice.loser_attributes)

full_data_frame.to_csv('full_data_frame_2.csv')

class RunModels():
    def __init__(self, full_data_frame, classifier):
        self.full_data_frame = full_data_frame
        self.msk = np.random.rand(len(self.full_data_frame)) < 0.8
        self.train = self.full_data_frame[self.msk]
        self.test = self.full_data_frame[~self.msk]
        self.classifier = classifier

    def NormalizeDataFrame(self):
        #self.normalized_df = self.full_data_frame
        ##self.normalized_df = self.normalized_df.drop(['winner'],
                                                    # axis=1)  # drops winner column from the normalized data frame
        self.train = self.train.drop(['winner'], axis = 1 )
        self.test = self.test.drop(['winner'], axis = 1)
        self.train_ids = self.train.Fight_ID
        self.test_ids = self.test.Fight_ID
        self.train = self.train.drop(['Fight_ID'], axis =1 )
        self.test = self.test.drop(['Fight_ID'], axis =1 )
        self.train_labels = np.asarray(self.train['classification'])
        self.test_labels = np.asarray(self.test['classification'])
        self.train = self.train.drop(['classification'], axis=1)  # train dataframe dropping classification 1,0
        self.test = self.test.drop(['classification'], axis=1)

        scaler = MinMaxScaler()
        self.train_normalised = pd.DataFrame(scaler.fit_transform(self.train),
                                     columns=self.train.columns)  # normalising the columns
        self.test_normalised = pd.DataFrame(scaler.fit_transform(self.test),
                                     columns=self.test.columns)  # normalising the columns
        #self.normalized_df.to_csv('normalized_df.csv')

    def TrainAndTest(self):

        #self.train = self.train + np.random.rand(*self.train.shape)/1
        #print(self.train)
        self.train_normalised.to_csv('training_set.csv')
        self.train_normalised = self.train.values.tolist()
        self.test_normalised = self.test.values.tolist()

    def FitSVM(self):
        clf = svm.SVC(kernel='linear')
        clf.fit(self.train, self.train_labels)
        print(
        "SVM accuracy is", float(np.sum(clf.predict(self.test_normalised) == self.test_labels)) / len(self.test_labels))

    def FitNeural(self):
        clf_neural = self.classifier
        clf_neural.fit(self.train, self.train_labels)
        print("Neural network accuracy is:",
              float(np.sum(clf_neural.predict(self.test_normalised) == self.test_labels)) / len(self.test_labels))
        print(clf_neural.score(self.train, self.train_labels))

        # with open('my_dumped_classifier.pkl', 'wb') as fid:
        #     pickle.dump(clf_neural, fid)

Model = RunModels(full_data_frame, gnb_loaded)

Model.NormalizeDataFrame()
Model.TrainAndTest()
#Model.FitSVM()
Model.FitNeural()
