# Collaborative-Filtering-Model
An implementation of memory-based collaborative filtering on a food recipe dataset using the Vector Space Similarity to predict users' ratings of food dishes.
import numpy as np
import pandas as pd
import sys
import math
import scipy as sp
import gc
import json
import statistics
from collections import OrderedDict
import operator
import matplotlib.pyplot as plt
import scipy as sp


def main():

    # Initiate Command-Line arguments
    dishes = pd.read_csv(sys.argv[1]).fillna(1.0)
    train = open(sys.argv[2],)
    train_dict = json.load(train)
    test = open(sys.argv[3],)
    test_dict = json.load(test)

    # Calculate MAE, Precision & Recall using Vector Space Similarity:
    run_test(dishes, test_dict, "v")

    # Calculate MAE, Precision & Recall using Pearson Correlation Coefficient Similarity:
    #run_test(dishes, test_dict, "p")

    modDfObj = dishes.append({'dish_id' : 1000 , 'dish_name' : 'new recipe'} , ignore_index=True)

    #print(modDfObj)

    return


def run_test(dishes, test_dict, flag):

    data = []

    # Calculate Mean Absolute Error
    mae = calculate_mae(dishes, test_dict, flag)
    print('Task 1 MAE:', end=' ')
    print(mae)

    # Calculate Precision & Recall for top 10, 20
    for k in [10, 20]:
        pr = calculate_precision_recall(dishes, test_dict, k, flag)
        print('Task 2 Precision@', end='')
        print(k, end=': ')
        print(pr[0])
    for k in [10, 20]:
        pr = calculate_precision_recall(dishes, test_dict, k, flag)
        data.append(pr)
        print('Task 2 Recall@', end='')
        print(k, end=': ')
        print(pr[1])


def calculate_vss(user1, user2): # Calculate Vector Space Similarity between 2 users

    # Obtain similar dishes reviewed between users
    similar_dishes = intersection(user1.keys(), user2.keys())

    # Calculate numerator
    numerator = 0.0
    for dish in similar_dishes:
        numerator += (user1[dish] * user2[dish])

    # Calculate denominator
    denominator = 0.0

    u1_var = 0.0
    for rating in user1.values():
        u1_var += math.pow(rating,2)

    u2_var = 0.0
    for rating in user2.values():
        u2_var += math.pow(rating,2)

    denominator = math.sqrt(u1_var) * math.sqrt(u2_var)

    # Return vector space similarity
    vss = numerator/denominator
    return vss


def calculate_pcc(user1, user2):

    # Obtain average ratings per user
    u1_avg = statistics.mean(user1.values())
    u2_avg = statistics.mean(user2.values())

    # Obtain similar dishes reviewed between users
    similar_dishes = intersection(user1.keys(), user2.keys())

    # Calculate numerator
    numerator = 0.0
    for dish in similar_dishes:
        numerator += (user1[dish] - u1_avg) * (user2[dish] - u2_avg)

    # Calculate denominator
    denominator = 0.0

    u1_var = 0.0
    for rating in user1.values():
        val = rating - u1_avg
        u1_var += math.pow(val,2)

    u2_var = 0.0
    for rating in user2.values():
        val = rating - u2_avg
        u2_var += math.pow(val,2)

    denominator = math.sqrt(u1_var) * math.sqrt(u2_var)

    # Return vector space similarity
    pcc = numerator/denominator
    return pcc


def predict_ratings(user1, dishes, df, flag): # Calculate memory-based predictions for a user
    # FORMULA: prediction = avg_ratings + SUM[(vss)*(user2_rating - avg_ratings)] / SUM(|vss|)

    # Variables
    predictions = dict()

    # Calculate user's average ratings
    avg_ratings = statistics.mean(user1.values())

    # Calculate Numerator
    for d in user1.keys(): # iterate through dishes
        sim_sum = 0.0
        numerator = 0.0
        for u in df.values(): # Iterate through users
            user2 = dict(u)
            if d not in user2.keys(): continue
            else:
                # Calculating VSS between user1 and user2
                if flag == "v": similarity = calculate_vss(user1, user2)
                else: similarity = calculate_pcc(user1, user2)
                sim_sum += similarity

                # Calculate Numerator:
                numerator += similarity * (user2[d] - avg_ratings)

        # Calculate Denominator:
        denominator = abs(sim_sum)

        # Enter predicted rating into user dictionary
        if sim_sum == 0:
            return predictions
        else:
            prediction = round(avg_ratings + (numerator/denominator)) * 1.0
            predictions[d] = prediction

    return predictions


def calculate_mae(dishes, df, flag):
    # FORMULA: MAE = SUM[(vss)*(user2_rating - avg_ratings)] / SUM(|vss|)

    numerator = 0.0
    for u in df.values():
        user = dict(u) # dishes = user.keys(), ratings = user.values()
        predictions = predict_ratings(user, dishes, df, flag)

        # Get the absolute value (val) of prediction - actual ratings
        val = 0.0
        for d in user.keys():
            val += (predictions[d] - user[d])
        numerator += abs(val)

    denominator = len(df)
    mae = numerator/denominator
    return mae


def calculate_precision_recall(dishes, df, k, flag):

    precision = []
    recall = []

    for u in df.values():
        user = dict(u) # dishes = user.keys(), ratings = user.values()
        predictions = predict_ratings(user, dishes, df, flag)
        real_recs, predicted_recs = dict(), dict()
        rel, rel_retrieved = 0.0, 0.0

        # Sort dictionaries by rank:
        temp = sorted(user.items(), key=operator.itemgetter(1), reverse=True)
        user = dict(temp)
        temp = sorted(predictions.items(), key=operator.itemgetter(1), reverse=True)
        predictions = dict(temp)

        # Recommend Dishes
        for dish in user.keys():
            if user[dish] >= 3: real_recs[dish] = 1
            else: real_recs[dish] = 0

            if predictions[dish] >= 3: predicted_recs[dish] = 1
            else: predicted_recs[dish] = 0

        # Calculate retrieved, relevant, and relevant retrieved dishes for each user based on rank (10,20)
        count1, count2 = 0, 0
        for r in real_recs.values():
            if r == 1: rel+=1
            count1 += 1
            if count1 == k: break
        for r in predicted_recs.values():
            if r == 1: rel_retrieved+=1
            count2 += 1
            if count2 == k: break

        # Obtain precision and recall for each user
        precision.append(rel_retrieved/len(user))
        recall.append(rel_retrieved/rel)


    return (statistics.mean(precision), statistics.mean(recall))


def intersection(lst1, lst2):
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


# Running Main
if __name__ == '__main__':
    main()% 
