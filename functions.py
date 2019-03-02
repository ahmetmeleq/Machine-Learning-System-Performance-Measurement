from math import isnan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve
import sys


def import_and_normalize_data(data_path='data1_SM.txt'):
    data = pd.read_csv(data_path, sep=",", header=None)
    data = data.values

    for i in range(len(data)):
        for j in range(len(data[i])):
            if isnan(data[i][j]):
                data[i][j] = 0


    max_value, max_index = max((x, (i, j))
                               for i, row in enumerate(data)
                               for j, x in enumerate(row))

    min_value, min_index = min((x, (i, j))
                               for i, row in enumerate(data)
                               for j, x in enumerate(row))


    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = ((data[i][j] - min_value) / (max_value - min_value))

    return data, max_value, min_value


def generate_label_matrix(labels_path='data1_Class_Labels.txt'):
    labels = pd.read_csv(labels_path, sep=",", header=None)
    labels = labels.values
    myarr = np.zeros((len(labels), len(labels)), dtype=bool)
    for i in range(len(labels)):
        for j in range(i, len(labels)):
            if labels[i] == labels[j]:
                myarr[i][j] = True
            else:
                myarr[i][j] = False

    return myarr


def find_frr_given_far(normalized_data, label_matrix, given_far=0.1, fudge_factor=0.00001,
                       init_threshold=0.0, steps=0.3):

    threshold = init_threshold
    far_minus_given_far_current_value = 1
    far_minus_given_far_last_value = 1

    counter_all = 0
    counter_impostor = 0
    counter_genuine = 0

    for i in range(len(label_matrix)):
        for j in range(i + 1, len(label_matrix)):
            if label_matrix[i][j] == True:
                counter_genuine = counter_genuine + 1
            elif label_matrix[i][j] == False:
                counter_impostor = counter_impostor + 1
            counter_all = counter_all + 1

    print('counter all', counter_all)
    print('impostors', counter_impostor)
    print('genuines', counter_genuine)

    while 1:

        counter_all = 0
        counter_fr = 0
        counter_fa = 0
        threshold = threshold + steps

        print("\n \n", 'New iteration')
        print("step size:", steps)

        for i in range(len(label_matrix)):
            for j in range(i + 1, len(label_matrix)):

                if (normalized_data[i][j] < threshold) & (label_matrix[i][j]==True):
                    counter_fr = counter_fr + 1
                elif (normalized_data[i][j] > threshold) & (label_matrix[i][j]==False):
                    counter_fa = counter_fa + 1
                counter_all = counter_all + 1
        try:
            far = counter_fa / counter_impostor
        except ZeroDivisionError:
            far = 0.999999999
            print('There was a zero on denominator. (far assignment)')

        far_minus_given_far_current_value = far - given_far
        print("Current fudge:", far_minus_given_far_current_value)
        print("Last fudge:", far_minus_given_far_last_value)

        if (far_minus_given_far_current_value * far_minus_given_far_last_value) <= 0:
            if abs(steps) < 0.0003:
                break
            else:
                steps = steps * (-1 / 10)



        if (abs(far_minus_given_far_current_value)<fudge_factor):
            break




        try:
            current_frr = counter_fr / counter_genuine
        except ZeroDivisionError:
            current_frr=0.999999999
            print('There was a zero on denominator. (current_frr assignment)')

        try:
            current_far = counter_fa / counter_impostor
        except ZeroDivisionError:
            current_far=0.999999999
            print('There was a zero on denominator. (current_far assignment)')


        print('this iter threshold is:', threshold)
        print('Current far', current_far)
        print('Current FRR', current_frr)
        print("next step size", steps)
        far_minus_given_far_last_value = far_minus_given_far_current_value

    return threshold, current_far, current_frr


def find_eer(normalized_data, label_matrix, init_threshold=0, steps=0.3):
    threshold = init_threshold
    counter_fa = 0
    counter_fr = 0

    print("\n \n", 'First iteration')
    print("step size:", steps)
    counter_all = 0
    for i in range(len(label_matrix)):
        for j in range(i + 1, len(label_matrix)):
            if ((normalized_data[i][j] < threshold) & ((label_matrix[i][j]) == True)):
                counter_fr = counter_fr + 1

            elif ((normalized_data[i][j] > threshold) & ((label_matrix[i][j]) == False)):
                counter_fa = counter_fa + 1

            counter_all = counter_all + 1
    fa_minus_fr_last_value = counter_fa - counter_fr
    fa_minus_fr_current_value = counter_fa - counter_fr
    print('threshold is:', threshold)
    print("Current value of False Acceptance minus False Rejection:", fa_minus_fr_current_value)
    print("Last value of False Acceptance minus False Rejection:", fa_minus_fr_last_value)
    threshold = threshold + steps
    counter_fr = 0
    counter_fa = 0

    while 1:
        print("\n \n", 'New iteration')
        print("step size:", steps)
        counter_all = 0
        for i in range(len(label_matrix)):
            for j in range(i + 1, len(label_matrix)):

                if ((normalized_data[i][j] < threshold) & ((label_matrix[i][j]) == True)) :
                    counter_fr = counter_fr + 1
                elif ((normalized_data[i][j] > threshold) & ((label_matrix[i][j]) == False)):
                    counter_fa = counter_fa + 1
                counter_all = counter_all + 1

        fa_minus_fr_current_value = counter_fa - counter_fr
        print("Current value of False Acceptance minus False Rejection:", fa_minus_fr_current_value)
        print("Last value of False Acceptance minus False Rejection:", fa_minus_fr_last_value)
        if fa_minus_fr_current_value * fa_minus_fr_last_value <= 0:
            if abs(steps) < 0.001:
                break

            if abs(fa_minus_fr_current_value) < 10:
                if counter_fa != 0:
                    break


            else:
                steps = steps * (-1 / 10)

        # 0.98501
        # 0.00001
        threshold = threshold + steps
        fa_minus_fr_last_value = fa_minus_fr_current_value
        counter_fr = 0
        counter_fa = 0

        print('threshold is:', threshold)

    return threshold, fa_minus_fr_current_value, fa_minus_fr_last_value, counter_fa, counter_fr, counter_all


def denormalize_threshold(threshold, max_value, min_value):
    threshold = threshold * (max_value - min_value) + min_value
    return threshold


def denormalize_data(data, max_value, min_value):

    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = data[i][j] * (max_value-min_value) + min_value

    return data


# for graphing purposes
def flatten(data):
    counter = 0
    for i in range(len(data)):
        for j in range(i+1,len(data[i])):
            counter += 1
    myvec = np.zeros((counter))
    counter2 = 0

    for i in range(len(data)):
        for j in range(i+1,len(data[i])):
            myvec[counter2] = data[i][j]
            counter2 += 1
    return myvec

# for graphing purposes, aggregates genuine examples into one vector, aggregates impostors to another.
def seperator (flat_data,flat_label_matrix):
    true_count, false_count = count_true_false(flat_label_matrix)
    geniuine_vector = np.zeros((true_count))
    impostor_vector = np.zeros((false_count))

    counter_1 = 0
    counter_2 = 0

    for item in range(1,len(flat_data)):
        if flat_label_matrix[item] == True:
            geniuine_vector[counter_1] = flat_data[item]
            counter_1 += 1
        elif flat_label_matrix[item] == False:
            impostor_vector[counter_2] = flat_data[item]
            counter_2 += 1
        else:
            print('Warning: Seperator has detected a non bool object. '
                  'Method: seperator might not be functioning correctly with this data.')

    return geniuine_vector,impostor_vector

# is used in seperator function, takes a 2D label matrix, transforms it into a 1D vector.
def count_true_false(flat_label_matrix):
    counter_true = 0
    counter_false = 0
    counter = 0
    for item in flat_label_matrix:

        if item == True:
            counter_true += 1

        if item == False:
            counter_false += 1

        if (item != False) and (item != True):
            print('Warning: True false counter has detected a non bool object. '
                  'Method: count_true_false might not be functioning correctly with this data.')
        counter += 1

    return counter_true, counter_false








# OUTPUT FUNCTIONS: These functions write the output logs on a file or saves graph on a file.
# roc_curve_draw does not save the graph. Saving is handled at ___main____ for roc curves. Saving process for other
# curves is handled within the functions.


def EER_output(data_p='data1_SM.txt', labels_p='data1_Class_Labels.txt',outputname='EER_output_1'):

    sys.stdout = open(outputname + '.txt', 'w')

    print('ANALYSIS ON:', data_p, 'STARTS')

    labelMatrix = generate_label_matrix(labels_path=labels_p)

    normalizedData, max_value, min_value = import_and_normalize_data(data_path=data_p)

    threshold, fa_minus_fr_current_value, fa_minus_fr_last_value, counter_fa, counter_fr, counter_all \
        = find_eer(normalizedData, labelMatrix)

    threshold = denormalize_threshold(threshold, max_value, min_value)
    eer1 = counter_fr / counter_all
    eer2 = counter_fa / counter_all
    error_rate = eer1 - eer2

    print('threshold:', threshold)
    print('eer:', eer1)
    print('fudge factor on eer is:', error_rate)
    sys.stdout.close()



def FAR_given_FRR_output(data_p='data1_SM.txt', labels_p='data1_Class_Labels.txt',outputname='EER_output_1',given_far=0.1):
    sys.stdout = open(outputname + '.txt', 'w')


    print('ANALYSIS ON:', data_p, 'STARTS')

    labelMatrix = generate_label_matrix(labels_path=labels_p)
    normalizedData, maxValue, minValue = import_and_normalize_data(data_path=data_p)

    threshold, current_far, current_frr = find_frr_given_far(normalizedData,
                                                             labelMatrix,
                                                             given_far=given_far,
                                                             fudge_factor=0.0000001,
                                                             init_threshold=0.1,
                                                             steps=0.1)
    print('\n')
    print('Threshold', threshold)
    print('FAR', current_far)
    print('FRR', current_frr)
    sys.stdout.close()



def plot(data_p='data1_SM.txt', labels_p='data1_Class_Labels.txt',save_name='myplot'):


    labelMatrix = generate_label_matrix(labels_path=labels_p)
    normalizedData, maxValue, minValue = import_and_normalize_data(data_path=data_p)
    data = denormalize_data(normalizedData, maxValue, minValue)
    data = flatten(data)
    labelMatrix_flat = flatten(labelMatrix)
    genuine_vector, impostor_vector = seperator(data,labelMatrix_flat)
    sns.distplot(genuine_vector,kde=True, hist=True, color='green')
    sns.distplot(impostor_vector,kde=True, hist=True, color='red')

    plt.savefig(save_name)
    plt.clf()



# uses scikit learn's lineplot and scatterplot functions to plot roc curves.
def roc_curve_draw(datap, labelp, color, label, plottype):

    if plottype=='scatterplot':

        # change these paths to evaluate with different data
        data_p = datap
        labels_p = labelp

        labelMatrix = generate_label_matrix(labels_path=labels_p)
        normalizedData, maxValue, minValue = import_and_normalize_data(data_path=data_p)
        data = denormalize_data(normalizedData, maxValue, minValue)
        data = flatten(data)
        labelMatrix_flat = flatten(labelMatrix)

        # roc_curve is imported from scikit-learn
        far, one_minus_frr, thresholds = roc_curve(labelMatrix_flat, data)
        sns.scatterplot(far, one_minus_frr, color=color, label=label)

    elif plottype=='lineplot':

        # change these paths to evaluate with different data
        data_p = datap
        labels_p = labelp

        labelMatrix = generate_label_matrix(labels_path=labels_p)
        normalizedData, maxValue, minValue = import_and_normalize_data(data_path=data_p)
        data = denormalize_data(normalizedData, maxValue, minValue)
        data = flatten(data)
        labelMatrix_flat = flatten(labelMatrix)

        # roc_curve is imported from scikit-learn
        far, one_minus_frr, thresholds = roc_curve(labelMatrix_flat, data)
        sns.lineplot(far, one_minus_frr, color=color, label=label)


    else:
        print('Error: Please set plottype parameter as either scatterplot or lineplot.')


def find_eer_2(normalized_data, label_matrix):

    counter_fa = 0
    counter_fr = 0

    normalized_data = np.sort(flatten(normalized_data))
    label_matrix = np.sort(flatten(label_matrix))

    len1 = len(normalizedData)

    index_1 = int(len1/2)


    threshold = normalizedData[index_1]

    print("\n \n", 'First iteration')
    counter_all = 0
    for i in range(len(label_matrix)):
        for j in range(i + 1, len(label_matrix)):

            if ((normalized_data[i][j] < threshold) & ((label_matrix[i][j]) == True)):
                counter_fr = counter_fr + 1
            elif ((normalized_data[i][j] > threshold) & ((label_matrix[i][j]) == False)):
                counter_fa = counter_fa + 1
            counter_all = counter_all + 1

    fa_minus_fr_last_value = counter_fa - counter_fr
    fa_minus_fr_current_value = counter_fa - counter_fr
    print('threshold is:', threshold)
    print("Current value of False Acceptance minus False Rejection:", fa_minus_fr_current_value)
    print("Last value of False Acceptance minus False Rejection:", fa_minus_fr_last_value)

    counter_fr = 0
    counter_fa = 0


    while 1:
        if(fa_minus_fr_current_value>0):
            index_1 = int((len1 + index_1)/2)
            threshold = normalized_data[index_1]

        elif(fa_minus_fr_current_value<0):
            index_1 = int(index_1/2)
            threshold = normalized_data[index_1]
        else:
            print('fa_minus_fr_current_value EQUALS ZERO')

        print("\n \n", 'New iteration')
        counter_all = 0
        for i in range(len(label_matrix)):
            for j in range(i + 1, len(label_matrix)):

                if ((normalized_data[i][j] < threshold) & ((label_matrix[i][j]) == True)) :
                    counter_fr = counter_fr + 1
                elif ((normalized_data[i][j] > threshold) & ((label_matrix[i][j]) == False)):
                    counter_fa = counter_fa + 1
                counter_all = counter_all + 1
