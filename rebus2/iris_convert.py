import csv
import sys

feature_labels = ['A', 'B', 'C', 'D']

division_parameter = 10
overlap_params1 = [5]
coefficient_power = 3

for ind_op, overlap_param1 in enumerate(overlap_params1):
    iris_name = {}
    coefficient_for_features = {}
    coef1_max = 1
    coef1_cut = coef1_max / (overlap_param1+1.0)

    print 'reading...'
    with open('iris.data', 'rb') as csvfile:
        iris_reader = csv.reader(csvfile, delimiter=',')
        iris_for_features = {}
        for idx, row in enumerate(iris_reader):
            iris_name[idx+1] = row[4]
            for idx_fl, feature_label in enumerate(feature_labels):
                # assign neighborhood coefficients for 1 decimal digits
                for neighborhood in range(-overlap_param1,1+overlap_param1):
                    feature_val = round(float(row[idx_fl]),1)+neighborhood/float(division_parameter)
                    feature = feature_label + (":%.1f" % feature_val)
                    coef1 = (coef1_max - coef1_cut * abs(neighborhood))**coefficient_power;
                    if iris_for_features.has_key(feature) == False:
                        iris_for_features[feature] = []
                        coefficient_for_features[feature] = []
                    iris_for_features[feature].append(idx+1)
                    coefficient_for_features[feature].append(coef1)

                
    print 'read.'

    print iris_for_features.keys()

    print 'writing...'
    f = open('iris'+str(division_parameter)+'-'+str(overlap_param1)+'-'+str(coefficient_power)+'.txt','w')
    for iris in iris_for_features.values():
        for iris1 in iris:
            f.write(iris_name[iris1]+'_'+str(iris1)+' ')
        f.write('\n')
    f.close()
    f = open('iris'+str(division_parameter)+'-'+str(overlap_param1)+'-'+str(coefficient_power)+'-element-weights.txt','w')
    for coef in coefficient_for_features.values():
        for coef1 in coef:
            f.write(str(coef1)+' ')
        f.write('\n')
    f.close()
    f = open('iris'+str(division_parameter)+'-'+str(overlap_param1)+'-'+str(coefficient_power)+'-features.txt','w')
    for feature in iris_for_features.keys():
        f.write(str(feature)+'\n')
    f.close()
    print 'written.'
    
