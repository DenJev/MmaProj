import csv
import pandas as pd

full_data_frame = pd.read_csv("data.csv", encoding = 'GBK')

def FighterWeight(weight, weight_fights , file_name):  # Generates CSV files for all fighters in certain weight class
    # necessary to run other file
    fights = full_data_frame[full_data_frame.B_Weight == weight]  # dataframe with certain weight class
    fights.to_csv(weight_fights)  # Outputs csv file with only weight class
    fighters = pd.DataFrame()  # Initialise empty dataframe for fighter names
    fighters['fighter_name'] = fights['B_Name']  # These two lines takes names
    fighters.append(fights['R_Name'])
    fighters.drop_duplicates(inplace = True)
    fighters = fighters.reset_index(drop = True)
    fighters.to_csv(file_name)  # Sends fighter names of weightclass to file

    with open(file_name) as f:
        for line in csv.reader(f, delimiter=','):
            individual_fighter = pd.DataFrame()  # Initialise empty dataframe
            individual_fighter = fights[fights.B_Name == line[1]]  # Append fights for each fighter
            individual_fighter = individual_fighter.append(fights[fights.R_Name == line[1]])
            individual_fighter.to_csv(line[1] + '.csv')  # Output csv file containing each fighters fights

FighterWeight(65, 'featherweight_fights.csv', 'featherweight_fighters.csv')
FighterWeight(70, 'lightweight_fights.csv', 'lightweight_fighters.csv')
FighterWeight(77, 'welterweight_fights.csv', 'welterweight_fighters.csv')



