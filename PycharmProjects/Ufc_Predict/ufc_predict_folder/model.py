import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
# noinspection PyPep8
import string

from sklearn.neural_network import MLPClassifier


class fighter:
    def __init__(self, name):
        self.name = name

    def fighter_type(self):
        lol = list(csv.reader(open('fighter_type.csv', 'rb'), delimiter='\t'))

        for i in range(1, len(lol)):
            if self.name == lol[i][0]:
                # print(lol[i][1])
                return lol[i][1]


class fights(fighter):

    def __init__(self, name, *args, **kwargs):
        fighter.__init__(self, name, *args, **kwargs)

    # print(self.name)

    def number_of_fights(self):
        count = 0
        with open(self.name + '.csv', 'r') as fight_file:
            for row in fight_file:
                count += 1

        print 'The number of fights', self.name, 'has been in is', count

        return count

    def filtered_fights(self, fight_result_type):
        self.fight_result_type = fight_result_type
        self.filtered_fights_array = pd.read_csv(self.name + '.csv')    # reads in csv file by fighter name
                                                                        # and stores as pandas dataframe
        self.filtered_fights_array = self.filtered_fights_array[self.filtered_fights_array.winby == fight_result_type]
        # filters pandas date frame for fight result
        self.filtered_fights_array.to_csv(self.name + fight_result_type + '.csv')

        return self.filtered_fights_array

    # writes dataframe to csv file containing only fights with allocated fight results

    def filtered_fights_three_round(self):  # use after filtered_fights activated

        self.filtered_fights_array = self.filtered_fights_array.reset_index(drop=True)

        # print(self.filtered_fights_array.B__Round4_Grappling_Reversals_Landed)

        self.filtered_fights_array = self.filtered_fights_array[
            self.filtered_fights_array.B__Round4_Grappling_Reversals_Landed.isna() == True]

        self.filtered_fights_array = self.filtered_fights_array[
            self.filtered_fights_array.R__Round4_Grappling_Reversals_Landed.isna() == True]

        # print(self.filtered_fights_array.R__Round3_Grappling_Reversals_Landed)

        self.filtered_fights_array = self.filtered_fights_array[
            self.filtered_fights_array.R__Round3_Grappling_Reversals_Landed.isna() == False]

        self.filtered_fights_array = self.filtered_fights_array[
            self.filtered_fights_array.B__Round3_Grappling_Reversals_Landed.isnull() == False]

        # print self.filtered_fights_array
        self.filtered_fights_array.to_csv(self.name + self.fight_result_type + '3_round.csv')

        return self.filtered_fights_array


class fight_manipulation(fights):

    def __init__(self, name, *args, **kwargs):
        fighter.__init__(self, name, *args, **kwargs)

    def winner(self):

        winner_of_fight = self.filtered_fights_array.winner     # defining winner of fight, red or blue
        self.chosen_attributes = pd.DataFrame()                 # empty dataframe
        self.loser_attributes = pd.DataFrame()                  # Loser attributes
        self.chosen_attributes['winner'] = winner_of_fight      # adds winner of fights to dataframe
        self.loser_attributes['winner'] = np.where(self.filtered_fights_array.winner == 'blue', 'red', 'blue')
        self.chosen_attributes['classification'] = 1
        self.loser_attributes['classification'] = -1
        self.chosen_attributes = self.chosen_attributes.reset_index(drop=True)  # reset index: 0,1,2,3,4...
        # return (self.loser_attributes, self.chosen_attributes)

    def attribute_difference(self, string):

        IsFightAttribute = False
        opposite_string = 'R' + string[1:]
        if '__' in string:
            IsFightAttribute = True
        else:
            IsFightAttribute = False

        diff_blue = self.filtered_fights_array[string] - self.filtered_fights_array[opposite_string]
        diff_red = self.filtered_fights_array[opposite_string] - self.filtered_fights_array[string]
        self.chosen_attributes[string] = np.where(self.filtered_fights_array.winner == 'blue', diff_blue, diff_red)
        self.loser_attributes[string] = np.where(self.filtered_fights_array.winner == 'blue', diff_red, diff_blue)


    def add_fighter_type(self):

        self.chosen_attributes['Striker'] = 0
        self.chosen_attributes['Wrestler'] = 0
        self.chosen_attributes['Bjj'] = 0
        self.chosen_attributes['NA'] = 0
        self.chosen_attributes['Judo'] = 0
        self.filtered_fights_array = self.filtered_fights_array.reset_index(drop=True)

        self.loser_attributes['Striker'] = 0
        self.loser_attributes['Wrestler'] = 0
        self.loser_attributes['Bjj'] = 0
        self.loser_attributes['NA'] = 0
        self.loser_attributes['Judo'] = 0

        for i in self.filtered_fights_array.index:
            if self.filtered_fights_array.at[i, 'winner'] == 'red':
                winner_type = fighter(self.filtered_fights_array.at[i, 'R_Name']).fighter_type()
                loser_type = fighter(self.filtered_fights_array.at[i, 'B_Name']).fighter_type()

            elif self.filtered_fights_array.at[i, 'winner'] == 'blue':
                winner_type = fighter(self.filtered_fights_array.at[i, 'B_Name']).fighter_type()
                loser_type = fighter(self.filtered_fights_array.at[i, 'R_Name']).fighter_type()

            if winner_type == 'Striker':
                self.chosen_attributes.at[i, 'Striker'] = 1
            elif winner_type == 'Bjj':
                self.chosen_attributes.at[i, 'Bjj'] = 1
            elif winner_type == 'NA':
                self.chosen_attributes.at[i, 'NA'] = 1
            elif winner_type == 'Judo':
                self.chosen_attributes.at[i, 'Judo'] = 1
            elif winner_type == 'Wrestler':
                self.chosen_attributes.at[i, 'Wrestler'] = 1
            if loser_type == 'Striker':
                self.loser_attributes.at[i, 'Striker'] = 1
            elif loser_type == 'Bjj':
                self.loser_attributes.at[i, 'Bjj'] = 1
            elif loser_type == 'NA':
                self.loser_attributes.at[i, 'NA'] = 1
            elif loser_type == 'Judo':
                self.loser_attributes.at[i, 'Judo'] = 1
            elif loser_type == 'Wrestler':
                self.loser_attributes.at[i, 'Wrestler'] = 1
        return self.chosen_attributes


fighter_loop = list(csv.reader(open('FighterAndFighterType.csv', 'rb')))

full_data_frame = pd.DataFrame()

for i in range(1, len(fighter_loop) - 1):
    object_choice = fight_manipulation(fighter_loop[i][1])
    # print fighter_loop[i][1]
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

    # print object_choice.chosen_attributes

    full_data_frame = full_data_frame.append(object_choice.chosen_attributes)
    full_data_frame = full_data_frame.append(object_choice.loser_attributes)

full_data_frame.to_csv('full_data_frame.csv')


class RunModels():
    def __init__(self, full_data_frame):
        self.full_data_frame = full_data_frame

    def NormalizeDataFrame(self):
        print("hello")
        self.normalized_df = self.full_data_frame
        self.normalized_df = self.normalized_df.drop(['winner'],
                                                     axis=1)  # drops winner column from the normalized data frame
        scaler = MinMaxScaler()
        self.normalized_df = pd.DataFrame(scaler.fit_transform(self.normalized_df),
                                     columns=self.normalized_df.columns)  # normalising the columns
        self.normalized_df.to_csv('normalized_df.csv')

    def TrainAndTest(self):
        print("hello")
        msk = np.random.rand(len(self.normalized_df)) < 0.8
        self.train = self.normalized_df[msk]  # train data frame
        self.train_labels = np.asarray(self.train['classification'])  # training classification labels list
        self.train = self.train.drop(['classification'], axis=1)  # train dataframe dropping classification 1,0
        #self.train = self.train + np.random.rand(*self.train.shape)/1
        #print(self.train)
        self.train.to_csv('training_set.csv')
        self.test = self.normalized_df[~msk]  # test dataframe
        self.test_labels = np.asarray(self.test['classification'])
        self.test = self.test.drop(['classification'], axis=1)
        self.train_attributes = self.train.values.tolist()
        self.test_attributes = self.test.values.tolist()

    def FitSVM(self):
        print("hello")
        clf = svm.SVC(kernel='linear')
        clf.fit(self.train, self.train_labels)
        print(
        "SVM accuracy is", float(np.sum(clf.predict(self.test_attributes) == self.test_labels)) / len(self.test_labels))

    def FitNeural(self):
        print("hello")
        clf_neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25, 12, 5), random_state=1)
        clf_neural.fit(self.train, self.train_labels)
        print("Neural network accuracy is:",
              float(np.sum(clf_neural.predict(self.test_attributes) == self.test_labels)) / len(self.test_labels))
        print(clf_neural.score(self.train, self.train_labels))


Model = RunModels(full_data_frame)

Model.NormalizeDataFrame()
Model.TrainAndTest()
Model.FitSVM()
Model.FitNeural()
