# Script to classify the low-throughput data with a random forest
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn import preprocessing
import sys
import itertools
from sklearn import metrics



######   Tuning parameters   ######
train_size = .8
######   Function definitiions   ######
# Function to separate ternaries by name
def ternaryName(row,elements):
    composition = row[elements].nonzero()[0]
    return '-'.join(sorted(elements[composition].values))

def write_for_plotting(mydata,pred,filename,elems):
    mydata['formula'] = ''
    for i,x in mydata.iterrows():
        tn = x['ternary_name']
        # elems = tn.split('-')
        # elems = ['Al','Ni','Zr']
        f = ''
        for e in elems:
            f = f+e
            f = f+str(x[e])
        # print f
        mydata.at[i,'formula'] = f


    with open(filename,'w') as writefile:
        mydata.reset_index()
        for i in np.arange(mydata.shape[0]):
        # for i,x in mydata.iterrows():
            x = mydata.iloc[[i]]
            p = pred[i][1]
            # f = mydata.at[i,'formula']
            f = x['formula'].values[0]
            elems = ["".join(x) for _, x in itertools.groupby(f, key=str.isalpha)]
            elemline = ' '.join(elems)
            writefile.write('{} {:.4f}\n'.format(elemline,p))

def load_prep_data(filename,allfeatures):
    ######   Load data to predict from ../data/comp_and_features.csv   ######
    mydata = pd.read_csv(filename,header=0)
    del mydata['gfa_measured'] #remove feature that is equivalent to class
    myfeatures = mydata.columns[:-2]
    myend_of_elem = np.where(myfeatures.values=='NComp')[0][0]
    # Extract list of elements (elements)
    myelements = myfeatures[0:myend_of_elem]
    mydata['is_train'] = 0
    mydata['ternary_name'] = mydata.apply(lambda row: ternaryName(row,myelements),axis=1)

    # populate elements that this data didn't have
    allmyFeatures = list(mydata)
    missing_features = [x for x in allfeatures if x not in allmyFeatures]

    for f in missing_features:
        mydata[f] = 0

    mydata[not_elements] = preprocessing.scale(mydata[not_elements])

    return mydata

def real_class_to_proba(mydata,elemlist):
    # return the real class values in the same form as clf.predict_proba
    # also can select specific ternaries and sub-ternaries
    data_sel = mydata[mydata.ternary_name.isin(elemlist)]
    data_val = data_sel.as_matrix(columns=['Class'])
    data_val = list(data_val.flatten())
    data_list = np.array(zip(data_val,[-(x-1) for x in data_val]))

    return data_list

def split_test_train(data, ternaries, percentTest, splitHTP=True):

    train_size = 1.0-percentTest
    #for j in range(100):
    # Split data into training and test sets, based on train_size parameter
    tern_shuf = list(zip(ternaries[0],ternaries[1]))
    np.random.shuffle(tern_shuf)

    # Move ternaries of interest to test set
    if splitHTP:
        temp = [tern_shuf.index(item) for item in tern_shuf if
                (item[0] == 'Al-Ni-Zr' or item[0] == 'Co-V-Zr' or 
                 item[0] == 'Co-Fe-Zr' or item[0] == 'Fe-Nb-Ti' or 
                 item[0] == 'Al-Ni' or item[0] == 'Al-Zr' or item[0] == 'Ni-Zr' or 
                 item[0] == 'Co-V' or item[0] == 'Co-Fe' or item[0] == 'Co-Zr' or 
                 item[0] == 'V-Zr' or item[0] == 'Fe-Zr')]
        for i in range(len(temp)):
            tern_shuf.append(tern_shuf.pop(temp[i]-i))

    # Split training and test sets
    ternaries = [list(t) for t in zip(*tern_shuf)]
    tern_train = np.extract(np.cumsum(ternaries[1])/sum(ternaries[1])<=train_size,
                            ternaries)
    data['is_train'] = data['ternary_name'].isin(tern_train)
    # To use min/max error test and train sets, uncomment following lines
    # min_train = np.loadtxt('min_error_train.txt',dtype='str')
    # data['is_train'] = data['ternary_name'].isin(min_train)
    # To use randomly chosen data, rather than data separated by ternaries, 
    # uncomment this line
    #data['is_train'] = np.random.uniform(0,1,len(data))<=train_size
    train, test = data[data['is_train']==True], data[data['is_train']==False]
    print('Number of observations in the training data:',len(train))
    print('Number of observations in the test data:',len(test))
    # Build classification vector (y)
    y = train['Class'].astype(int)

    return train, test, y


def plot_learning_curve(train,test,filename,plot=True):
    ######   Plot learning curve   ######
    # features = imp_feat
    train_sizes = []
    train_scores = []
    train_err = []
    test_scores = []
    train_subset_scores = []
    test_err = []
    subset_size = [.01,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
    # subset_size = [0.1, 0.5, 1]
    for subset in subset_size:
        print('Training subset size: {}'.format(subset))
        
        train_subset = train[np.random.uniform(0,1,len(train))<=subset]
        train_sizes.append(len(train_subset))

        # clf = RandomForestClassifier(n_estimators=500,n_jobs=-1,oob_score=True)
        clf = MLPClassifier(activation='logistic')
        clf.fit(train_subset[features],train_subset['Class'].astype(int))

        test_val = test['Class'].astype(int).values
        test_proba = clf.predict_proba(test[features])
        test_ll = metrics.log_loss(test_val,test_proba[:,1])
        test_scores.append(test_ll)

        train_subset_val = train_subset['Class'].astype(int).values
        train_subset_proba = clf.predict_proba(train_subset[features])
        train_subset_ll = metrics.log_loss(train_subset_val,train_subset_proba[:,1])
        train_subset_scores.append(train_subset_ll)

        train_val = train['Class'].astype(int).values
        train_proba = clf.predict_proba(train[features])
        train_ll = metrics.log_loss(train_val,train_proba[:,1])
        train_scores.append(train_ll)

        # train_scores.append(clf.oob_score_)
        # test_pred = clf.predict(test[features])
        # test_scores.append(clf.score(test[features],test_val))
        del train_subset
        del clf


    if plot:
        plt.plot(subset_size,train_subset_scores,'c',linestyle='--', marker='o')
        plt.plot(subset_size,train_scores,'b',linestyle='--', marker='o')
        plt.plot(subset_size,test_scores,'r',linestyle='--', marker='o')
        plt.xlabel('% of training samples')
        plt.ylabel('Log loss')
        plt.legend(['Actual training set','Full Training data','Test set'])
        plt.savefig(filename,dpi=200)
        plt.show()
        # plt.clf()
        # plt.savefig('/Users/brenna/Desktop/Git/CS229/LCnn/Learning_curve_LBonly.png')
        
    

    return subset_size,train_scores,test_scores,train_subset_scores


def plot_learning_curve_HTP(LBtrain,HTPtrain,test,HTPtest,filename,plot=True):
    ######   Plot learning curve   ######
    # features = imp_feat
    train_sizes = []
    train_scores = []
    train_err = []
    test_scores = []
    test_err = []
    HTPtest_scores = []
    # subset_size = [0,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
    subset_size = [0,0.0001,0.001,0.005, 0.01,0.02,0.03, 0.05,0.075, 0.1,0.15,0.2,0.3, 0.5,0.75, 1]
    # subset_size = [0,0.001,0.005, 0.01, 0.1, 1]
    for subset in subset_size:
        # print('Training subset size: {}'.format(subset))
        
        HTPtrain_subset = HTPtrain[np.random.uniform(0,1,len(HTPtrain))<=subset]
        HTPtrain_size = len(HTPtrain_subset)
        train_sizes.append(len(HTPtrain_subset))
        print('Training subset size: {}% = {}'.format(subset*100,HTPtrain_size))

        train_subset = pd.concat([LBtrain, HTPtrain_subset])

        # clf = RandomForestClassifier(n_estimators=500,n_jobs=-1,oob_score=True)
        clf = MLPClassifier(activation='logistic')
        clf.fit(train_subset[features],train_subset['Class'].astype(int))

        test_val = test['Class'].astype(int).values
        test_proba = clf.predict_proba(test[features])
        test_ll = metrics.log_loss(test_val,test_proba[:,1])
        test_scores.append(test_ll)

        HTPtest_val = HTPtest['Class'].astype(int).values
        HTPtest_proba = clf.predict_proba(HTPtest[features])
        HTPtest_ll = metrics.log_loss(HTPtest_val,HTPtest_proba[:,1])
        HTPtest_scores.append(HTPtest_ll)

        train_val = LBtrain['Class'].astype(int).values
        train_proba = clf.predict_proba(LBtrain[features])
        train_ll = metrics.log_loss(train_val,train_proba[:,1])
        train_scores.append(train_ll)

        # train_scores.append(clf.oob_score_)
        # test_pred = clf.predict(test[features])
        # test_scores.append(clf.score(test[features],test_val))
        del train_subset
        del clf


    if plot:
        plt.plot(subset_size,train_scores,'b',linestyle='--', marker='o')
        plt.plot(subset_size,test_scores,'r',linestyle='--', marker='o')
        plt.plot(subset_size,HTPtest_scores,'g',linestyle='--', marker='o')
        plt.xlabel('% of HTP data')
        plt.ylabel('Log loss')
        plt.legend(['Training set','Test set','HTP Test set'])
        # plt.clf()
        # plt.savefig('/Users/brenna/Desktop/Git/CS229/LCnn/Learning_curve_LBonly.png')
        plt.savefig(filename,dpi=200)
        plt.show()
    

    return subset_size,train_scores,test_scores, list(HTPtest_scores)


#--------------------------------------

######   Load low-throughput data from ../data/comp_and_features.csv   ######
data = pd.read_csv('../data/comp_and_features.csv',header=0)
del data['gfa_measured'] #remove feature that is equivalent to class
# Extract list of feature names (features)
features = data.columns[:-2]
end_of_elem = np.where(features.values=='NComp')[0][0]
# Extract list of elements (elements)
elements = features[0:end_of_elem]
not_elements = features[end_of_elem:]
# Normalize data to have zero-mean and unit variance
data[not_elements] = preprocessing.scale(data[not_elements])
# Extract list of ternaries (ternaries) 
data['ternary_name'] = data.apply(lambda row: ternaryName(row,elements),axis=1)
ternaries = np.unique(data['ternary_name'].values,return_counts=True)

allfeatures = list(data)
hitp_data = load_prep_data('../data/hitp_glass_data_featurized.csv',allfeatures)
hitp_train_data = hitp_data[~hitp_data['ternary_name'].isin(['Co-Fe-Zr','Co-V-Zr','Fe-Nb-Ti'])]

# select every nth row from the high throughput training data
# n = 2
hitp_train_data = hitp_train_data.reset_index()
# hitp_train_data = hitp_train_data.iloc[0::n,:]

train, test, y = split_test_train(data,ternaries,.1,False)

# add the (maybe downsampled) hitp data to the whole LB train data set
all_train_data = pd.concat([train, hitp_train_data])


filename = '/Users/brenna/Desktop/Git/CS229/LCnn/Learning_curve_LBonly.png'
# subset_size,train_scores,test_scores,train_subset_scores = plot_learning_curve(train,test,filename,plot=True)

# plt.semilogy(subset_size,train_subset_scores,'c',linestyle='--', marker='o')
# plt.semilogy(subset_size,train_scores,'b',linestyle='--', marker='o')
# plt.semilogy(subset_size,test_scores,'r',linestyle='--', marker='o')
# plt.xlabel('% of training samples')
# plt.ylabel('Log loss')
# plt.legend(['Actual training set','Full Training data','Test set'])
# plt.savefig(filename,dpi=200)
# plt.show()

# lb_lc_sizes = subset_size
# lb_lc_train = []
# lb_lc_test = []
# for i in range(len(subset_size)):
#     lb_lc_train.append([train_scores[i]])
#     lb_lc_test.append([test_scores[i]])


# for i in range(4):
#     train, test, y = split_test_train(data,ternaries,.1,False)
#     filename = '/Users/brenna/Desktop/Git/CS229/LCnn/Learning_curve_LBonly.png'
#     subset_size,train_scores,test_scores = plot_learning_curve(train,test,filename,plot=False)
    
#     for i in range(len(subset_size)):
#         lb_lc_train[i].append(train_scores[i])
#         lb_lc_test[i].append(test_scores[i])


# lb_lc_train_avg = np.mean(lb_lc_train,axis = 1)
# lb_lc_train_std = np.std(lb_lc_train,axis = 1)

# lb_lc_test_avg = np.mean(lb_lc_test,axis = 1)
# lb_lc_test_std = np.std(lb_lc_test,axis = 1)

# plt.errorbar(subset_size,lb_lc_train_avg, yerr = lb_lc_train_std, fmt='--o')#,'b',linestyle='--', marker='o')
# plt.errorbar(subset_size,lb_lc_test_avg, yerr = lb_lc_train_std, fmt='--o')#,'r',linestyle='--', marker='o')
# plt.xlabel('% of training samples')
# plt.ylabel('Log loss')
# plt.legend(['Training set','Test set'])
# plt.savefig('/Users/brenna/Desktop/Git/CS229/LCnn/Learning_curve_LBonly_avg.png')

# plt.show()


##########HTP learning curve---------------------

hitp_train_data = hitp_data[~hitp_data['ternary_name'].isin(['Co-Fe-Zr','Co-V-Zr','Fe-Nb-Ti'])]
hitp_test_data = hitp_data[hitp_data['ternary_name'].isin(['Co-Fe-Zr','Co-V-Zr','Fe-Nb-Ti'])]
# select every nth row from the high throughput training data
# n = 2
hitp_train_data = hitp_train_data.reset_index()
hitp_test_data = hitp_test_data.reset_index()

train, test, y = split_test_train(data,ternaries,.1,False)

filename = '/Users/brenna/Desktop/Git/CS229/LCnn/Learning_curve_withHTP.png'
subset_size,train_scores,test_scores,HTPtest_scores = plot_learning_curve_HTP(train,hitp_train_data,test,hitp_test_data,filename,plot=True)

# plt.plot(subset_size,train_scores,'b',linestyle='--', marker='o')
plt.plot(subset_size,test_scores,'r',linestyle='--', marker='o')
plt.plot(subset_size,HTPtest_scores,'g',linestyle='--', marker='o')
plt.plot([subset_size[0],subset_size[-1]],[0.379,0.379],'k',linestyle=':')

plt.xlabel('% of HTP data')
plt.ylabel('Log loss')
# plt.legend(['Training set','Test set','HTP Test set'])
plt.legend(['Test set','HTP Test set'])
# plt.clf()
# plt.savefig('/Users/brenna/Desktop/Git/CS229/LCnn/Learning_curve_LBonly.png')
filename = '/Users/brenna/Desktop/Git/CS229/LCnn/Learning_curve_withHTP6.png'
plt.savefig(filename,dpi=200)
plt.show()



lb_lc_sizes = subset_size
lb_lc_train = []
lb_lc_test = []
htp_lc_test = []

for i in range(len(subset_size)):
    lb_lc_train.append([train_scores[i]])
    lb_lc_test.append([test_scores[i]])
    htp_lc_test.append([HTPtest_scores[i]])


for i in range(4):
    train, test, y = split_test_train(data,ternaries,.1,False)
    filename = '/Users/brenna/Desktop/Git/CS229/LCnn/Learning_curve_LBonly.png'
    subset_size,train_scores,test_scores,HTPtest_scores = plot_learning_curve_HTP(train,hitp_train_data,test,hitp_test_data,filename,plot=False)
    
    for i in range(len(subset_size)):
        lb_lc_train[i].append(train_scores[i])
        lb_lc_test[i].append(test_scores[i])
        htp_lc_test[i].append(HTPtest_scores[i])


lb_lc_train_avg = np.mean(lb_lc_train,axis = 1)
lb_lc_train_std = np.std(lb_lc_train,axis = 1)

lb_lc_test_avg = np.mean(lb_lc_test,axis = 1)
lb_lc_test_std = np.std(lb_lc_test,axis = 1)

htp_lc_test_avg = np.mean(htp_lc_test,axis = 1)
htp_lc_test_std = np.std(htp_lc_test,axis = 1)

htp_dict = {}
htp_dict['subset size'] = subset_size
htp_dict['LB Train avg'] = lb_lc_train_avg
htp_dict['LB Train std'] = lb_lc_train_std
htp_dict['LB Test avg'] = lb_lc_test_avg
htp_dict['LB Test std'] = lb_lc_test_std
htp_dict['HTP Test avg'] = htp_lc_test_avg
htp_dict['HTP Test std'] = htp_lc_test_std

pddict = pd.DataFrame.from_dict(htp_dict)
pddict.to_csv('/Users/brenna/Desktop/Git/CS229/LCnn/htp_learning_curve_data.csv')

print('subset size = {}'.format(subset_size))
print('LB Train avg = {}'.format(lb_lc_train_avg))
print('LB Train std = {}'.format(lb_lc_train_std))
print('LB Test avg = {}'.format(lb_lc_test_avg))
print('LB Test std = {}'.format(lb_lc_test_std))
print('HTP Test avg = {}'.format(htp_lc_test_avg))
print('HTP Test std = {}'.format(htp_lc_test_std))



plt.errorbar(subset_size,lb_lc_train_avg, yerr = lb_lc_train_std, fmt='--o')#,'b',linestyle='--', marker='o')
plt.errorbar(subset_size,lb_lc_test_avg, yerr = lb_lc_train_std, fmt='--o')#,'r',linestyle='--', marker='o')
plt.errorbar(subset_size,htp_lc_test_avg, yerr = htp_lc_test_std, fmt='--o')
plt.xlabel('% of HTP training data included')
plt.ylabel('Log loss')
plt.legend(['Training set','Test set','HTP Test set'])
plt.savefig('/Users/brenna/Desktop/Git/CS229/LCnn/Learning_curve_HTP_avg.png')

plt.show()

# htp_dict.print_to_csv('test_htp_vals.csv')


# plt.clf()
# plt.savefig('/Users/brenna/Desktop/Git/CS229/LCnn/Learning_curve_LBonly_avg.png')
# plt.clf()
# plt.savefig(filename)

# ######   Plot learning curve   ######
# # features = imp_feat
# train_sizes = []
# train_scores = []
# train_err = []
# test_scores = []
# test_err = []
# subset_size = [.01,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
# # subset_size = [0.1, 0.5, 1]
# for subset in subset_size:
#     print('Training subset size: {}'.format(subset))
    
#     train_subset = train[np.random.uniform(0,1,len(train))<=subset]
#     train_sizes.append(len(train_subset))

#     clf = RandomForestClassifier(n_estimators=500,n_jobs=-1,oob_score=True)
#     clf.fit(train_subset[features],train_subset['Class'].astype(int))

#     test_val = test['Class'].astype(int).values
#     test_proba = clf.predict_proba(test[features])
#     test_ll = metrics.log_loss(test_val,test_proba[:,1])
#     test_scores.append(test_ll)

#     train_val = train['Class'].astype(int).values
#     train_proba = clf.predict_proba(train[features])
#     train_ll = metrics.log_loss(train_val,train_proba[:,1])
#     train_scores.append(train_ll)

#     # train_scores.append(clf.oob_score_)
#     # test_pred = clf.predict(test[features])
#     # test_scores.append(clf.score(test[features],test_val))
#     del train_subset
#     del clf
# plt.plot(subset_size,train_scores,'b',linestyle='--', marker='o')
# plt.plot(subset_size,test_scores,'r',linestyle='--', marker='o')
# plt.xlabel('% of training samples')
# plt.ylabel('Log loss')
# plt.legend(['Training set','Test set'])
# plt.show()
# plt.clf()
# plt.savefig('/Users/brenna/Desktop/Git/CS229/LCnn/Learning_curve_LBonly.png')


#--------------------------------------

# ######   Load low-throughput data from ../data/comp_and_features.csv   ######
# data = pd.read_csv('../data/comp_and_features.csv',header=0)
# del data['gfa_measured'] #remove feature that is equivalent to class
# # Extract list of feature names (features)
# features = data.columns[:-2]
# end_of_elem = np.where(features.values=='NComp')[0][0]
# # Extract list of elements (elements)
# elements = features[0:end_of_elem]
# not_elements = features[end_of_elem:]
# # Normalize data to have zero-mean and unit variance
# data[not_elements] = preprocessing.scale(data[not_elements])
# # Extract list of ternaries (ternaries) 
# data['ternary_name'] = data.apply(lambda row: ternaryName(row,elements),axis=1)
# ternaries = np.unique(data['ternary_name'].values,return_counts=True)

# #min_error = 1
# #max_error = 0
# #for j in range(100):
# # Split data into training and test sets, based on train_size parameter
# tern_shuf = list(zip(ternaries[0],ternaries[1]))
# np.random.shuffle(tern_shuf)
# # Move ternaries of interest to test set
# temp = [tern_shuf.index(item) for item in tern_shuf if
#         (item[0] == 'Al-Ni-Zr' or item[0] == 'Co-V-Zr' or 
#          item[0] == 'Co-Fe-Zr' or item[0] == 'Fe-Nb-Ti' or 
#          item[0] == 'Al-Ni' or item[0] == 'Al-Zr' or item[0] == 'Ni-Zr' or 
#          item[0] == 'Co-V' or item[0] == 'Co-Fe' or item[0] == 'Co-Zr' or 
#          item[0] == 'V-Zr' or item[0] == 'Fe-Zr')]
# for i in range(len(temp)):
#     tern_shuf.append(tern_shuf.pop(temp[i]-i))
# # Split training and test sets
# ternaries = [list(t) for t in zip(*tern_shuf)]
# tern_train = np.extract(np.cumsum(ternaries[1])/sum(ternaries[1])<=train_size,
#                         ternaries)
# data['is_train'] = data['ternary_name'].isin(tern_train)
# # To use min/max error test and train sets, uncomment following lines
# # min_train = np.loadtxt('min_error_train.txt',dtype='str')
# # data['is_train'] = data['ternary_name'].isin(min_train)
# # To use randomly chosen data, rather than data separated by ternaries, 
# # uncomment this line
# #data['is_train'] = np.random.uniform(0,1,len(data))<=train_size
# train, test = data[data['is_train']==True], data[data['is_train']==False]
# print('Number of observations in the training data:',len(train))
# print('Number of observations in the test data:',len(test))
# # Build classification vector (y)
# y = train['Class'].astype(int)


# ######   Train Random Forest classifier   ######
# sys.stdout.write('Building Random Forest classifier......... ')
# # Create classifier
# clf = RandomForestClassifier(n_estimators=500,n_jobs=-1) #n_jobs: -1 runs on all avail cores
# # Train classifier on training data
# clf.fit(train[features],y)
# print('Done.')
# # Determine feature importance
# imp_feat = sorted(zip(clf.feature_importances_,train[features]))
# imp_feat = [list(t) for t in zip(*imp_feat)]
# imp_feat = imp_feat[1][-30:-1]


# ######   Test classifier on test data   ######
# # Predict classifications of test data
# test_pred = clf.predict(test[features])
# # Create vector with validation values of test data
# test_val = test['Class'].astype(int).values
# # Output the number of incorrect classifications
# print('Classifier generated ',np.sum(test_pred != test_val),
#       ' misclassifications out of ',len(test_val),' resulting in ',
#       np.sum(test_pred != test_val)/len(test_val),' classification error.')
#if (np.sum(test_pred != test_val)/len(test_val) < min_error):
#    train['ternary_name'].to_csv('min_error_train.txt',sep=',',index=False)
#    test['ternary_name'].to_csv('min_error_test.txt',sep=',',index=False)
#    min_error = np.sum(test_pred != test_val)/len(test_val)
#elif (np.sum(test_pred != test_val)/len(test_val) > max_error):
#    train['ternary_name'].to_csv('max_error_train.txt',sep=',',index=False)
#    test['ternary_name'].to_csv('max_error_test.txt',sep=',',index=False)
#    max_error = np.sum(test_pred != test_val)/len(test_val)

######   Output Al-Ni-Zr predictions   ######
# print(test[test['ternary_name'].isin(['Al-Ni-Zr','Al-Ni','Al-Zr','Ni-Zr'])])

######   Plot learning curve   ######
# features = imp_feat
# train_sizes = []
# train_scores = []
# test_scores = []
# for subset in [.01,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]:
#     clf = RandomForestClassifier(n_estimators=500,n_jobs=-1,oob_score=True)
#     train_subset = train[np.random.uniform(0,1,len(train))<=subset]
#     clf.fit(train_subset[features],train_subset['Class'].astype(int))
#     train_sizes.append(len(train_subset))
#     train_scores.append(clf.oob_score_)
#     test_pred = clf.predict(test[features])
#     test_scores.append(clf.score(test[features],test_val))
#     del train_subset
#     del clf
# plt.plot(train_sizes,train_scores,'b-')
# plt.plot(train_sizes,test_scores,'r-')
# plt.xlabel('# of training samples')
# plt.ylabel('OOB score')
# plt.legend(['Training set','Test set'])
# plt.show()

allfeatures = list(data)

# ####### add hitp dev data and retrain
# hitp_data = load_prep_data('../data/hitp_glass_data_featurized.csv',allfeatures)
# hitp_train_data = hitp_data[~hitp_data['ternary_name'].isin(['Co-Fe-Zr','Co-V-Zr','Fe-Nb-Ti'])]

# # select every nth row from the high throughput training data
# # n = 2
# hitp_train_data = hitp_train_data.reset_index()
# # hitp_train_data = hitp_train_data.iloc[0::n,:]

# # add the (maybe downsampled) hitp data to the whole LB train data set
# all_train_data = pd.concat([train, hitp_train_data])

# print('Number of observations in the training data:',len(all_train_data))
# print('Number of observations in the test data:',len(test))
# # Build classification vector (y)
# yall = all_train_data['Class'].astype(int)

# sys.stdout.write('\nBuilding Random Forest classifier......... \n')
# # Create classifier
# clf = RandomForestClassifier(n_estimators=500,n_jobs=-1) #n_jobs: -1 runs on all avail cores
# # Train classifier on training data
# clf.fit(all_train_data[features],yall)
# print('Done.')

# test_pred = clf.predict(test[features])
# test_proba = clf.predict_proba(test[features])
# # Create vector with validation values of test data
# test_val = test['Class'].astype(int).values
# # Output the number of incorrect classifications
# print('Classifier generated ',np.sum(test_pred != test_val),
#       ' misclassifications out of ',len(test_val),' resulting in ',
#       np.sum(test_pred != test_val)/len(test_val),' classification error.')

# logloss = metrics.log_loss(test_val,test_proba)



# ###########################
# #### Al Ni Zr ternary #####

# allfeatures = list(data)

# tri_alnizr = load_prep_data('../data/triangles_alnizr_featurized.csv',allfeatures)

# alnizr_pred = clf.predict_proba(tri_alnizr[features])

# write_for_plotting(tri_alnizr,alnizr_pred,'../data/alnizr_pred.csv',['Al','Ni','Zr'])

# alnizr_real = data[data.ternary_name.isin(['Al-Ni-Zr', 'Al-Ni','Ni-Zr','Al-Zr','Al','Ni','Zr'])]
# alnizr_real_val = alnizr_real.as_matrix(columns=['Class'])
# alnizr_real_val = list(alnizr_real_val.flatten())
# alnizr_real_val = np.array(zip(alnizr_real_val,[-(x-1) for x in alnizr_real_val]))

# write_for_plotting(alnizr_real,alnizr_real_val,'../data/alnizr_exp.csv',['Al','Ni','Zr'])


# #### all dev set ternaries for plotting #####
# dev_plot_pred = load_prep_data('../data/triangles_glass_featurized.csv',allfeatures)
# dev_plot_exp = load_prep_data('../data/hitp_glass_data_featurized.csv',allfeatures)

# # dev_plot_pred = clf.predict_proba(dev_plot_data)

# ###### Co Fe Zr ######
# cofezr_list = ['Co-Fe-Zr','Co-Fe','Co-Zr','Fe-Zr','Co','Fe','Zr']

# cofezr = dev_plot_pred[dev_plot_pred.ternary_name.isin(cofezr_list)]
# cofezr_pred = clf.predict_proba(cofezr[features])
# write_for_plotting(cofezr,cofezr_pred,'../data/cofezr_pred.csv',['Co','Fe','Zr'])

# cofezr_real = dev_plot_exp[dev_plot_exp.ternary_name.isin(cofezr_list)]
# cofezr_real_val = real_class_to_proba(cofezr_real,cofezr_list)

# write_for_plotting(cofezr_real,cofezr_real_val,'../data/cofezr_exp.csv',['Co','Fe','Zr'])

# ###### Co Fe Zr ######
# covzr_list = ['Co-V-Zr','Co-V','Co-Zr','V-Zr','Co','V','Zr']

# covzr = dev_plot_pred[dev_plot_pred.ternary_name.isin(covzr_list)]
# covzr_pred = clf.predict_proba(covzr[features])
# write_for_plotting(covzr,covzr_pred,'../data/covzr_pred.csv',['Co','V','Zr'])

# covzr_real = dev_plot_exp[dev_plot_exp.ternary_name.isin(covzr_list)]
# covzr_real_val = real_class_to_proba(covzr_real,covzr_list)

# write_for_plotting(covzr_real,covzr_real_val,'../data/covzr_exp.csv',['Co','V','Zr'])

# ###### Fe Nb Ti ######
# fenbti_list = ['Fe-Nb-Ti','Fe-Nb','Fe-Ti','Nb-Ti','Fe','Nb','Ti']

# fenbti = dev_plot_pred[dev_plot_pred.ternary_name.isin(fenbti_list)]
# fenbti_pred = clf.predict_proba(fenbti[features])
# write_for_plotting(fenbti,fenbti_pred,'../data/fenbti_pred.csv',['Fe','Nb','Ti'])

# fenbti_real = dev_plot_exp[dev_plot_exp.ternary_name.isin(fenbti_list)]
# fenbti_real_val = real_class_to_proba(fenbti_real,fenbti_list)

# write_for_plotting(fenbti_real,fenbti_real_val,'../data/fenbti_exp.csv',['Fe','Nb','Ti'])

