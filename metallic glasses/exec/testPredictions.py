# Script to classify the low-throughput data with a random forest
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn import preprocessing
import sys
import itertools
from sklearn import metrics
import random



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



#min_error = 1
#max_error = 0
#for j in range(100):
# Split data into training and test sets, based on train_size parameter
tern_shuf = list(zip(ternaries[0],ternaries[1]))

# plot ternary histogram
tern_n = [x[1] for x in tern_shuf]
bins = [0,4,10,25,50,100,205]
plt.figure()
plt.hist(tern_n,bins,edgecolor='black',facecolor = 'blue')
plt.title('Sparse Data Distribution')
plt.xlabel('Number of data points per ternary')
plt.ylabel('Number of ternaries')
plt.savefig('ternary_histogram.pdf')
plt.show()
print("Median ternary size is: {}\n".format(np.median(tern_n)))

np.random.shuffle(tern_shuf)


# Move ternaries of interest to test set
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


######   Train Random Forest classifier   ######
sys.stdout.write('Building Random Forest classifier......... ')
# Create classifier
clf = RandomForestClassifier(n_estimators=500,n_jobs=-1) #n_jobs: -1 runs on all avail cores
# Train classifier on training data
clf.fit(train[features],y)
print('Done.')
# Determine feature importance
imp_feat = sorted(zip(clf.feature_importances_,train[features]))
imp_feat = [list(t) for t in zip(*imp_feat)]
imp_feat = imp_feat[1][-30:-1]


######   Test classifier on test data   ######
# Predict classifications of test data
test_pred = clf.predict(test[features])
# Create vector with validation values of test data
test_val = test['Class'].astype(int).values
# Output the number of incorrect classifications
print('Classifier generated ',np.sum(test_pred != test_val),
      ' misclassifications out of ',len(test_val),' resulting in ',
      np.sum(test_pred != test_val)/len(test_val),' classification error.')
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

####### add hitp dev data and retrain
hitp_data = load_prep_data('../data/hitp_glass_data_featurized.csv',allfeatures)
hitp_train_data = hitp_data[~hitp_data['ternary_name'].isin(['Co-Fe-Zr','Co-V-Zr','Fe-Nb-Ti'])]

# select every nth row from the high throughput training data
# n = 2
hitp_train_data = hitp_train_data.reset_index()
# hitp_train_data = hitp_train_data.iloc[0::n,:]

# add the (maybe downsampled) hitp data to the whole LB train data set
all_train_data = pd.concat([train, hitp_train_data])

print('Number of observations in the training data:',len(all_train_data))
print('Number of observations in the test data:',len(test))
# Build classification vector (y)
yall = all_train_data['Class'].astype(int)

sys.stdout.write('\nBuilding Random Forest classifier......... \n')
# Create classifier
clf = RandomForestClassifier(n_estimators=500,n_jobs=-1) #n_jobs: -1 runs on all avail cores
# Train classifier on training data
clf.fit(all_train_data[features],yall)
print('Done.')

test_pred = clf.predict(test[features])
test_proba = clf.predict_proba(test[features])
# Create vector with validation values of test data
test_val = test['Class'].astype(int).values
# Output the number of incorrect classifications
print('Classifier generated ',np.sum(test_pred != test_val),
      ' misclassifications out of ',len(test_val),' resulting in ',
      np.sum(test_pred != test_val)/len(test_val),' classification error.')

logloss = metrics.log_loss(test_val,test_proba)



# ###########################
# #### Al Ni Zr ternary #####


tri_alnizr = load_prep_data('../data/triangles_alnizr_featurized.csv',allfeatures)

alnizr_pred = clf.predict_proba(tri_alnizr[features])

write_for_plotting(tri_alnizr,alnizr_pred,'../data/alnizr_pred.csv',['Al','Ni','Zr'])

alnizr_real = data[data.ternary_name.isin(['Al-Ni-Zr', 'Al-Ni','Ni-Zr','Al-Zr','Al','Ni','Zr'])]
# alnizr_real_val = alnizr_real.as_matrix(columns=['Class'])
# alnizr_real_val = list(alnizr_real_val.flatten())
# alnizr_real_val = np.array(zip(alnizr_real_val,[-(x-1) for x in alnizr_real_val]))

# write_for_plotting(alnizr_real,alnizr_real_val,'../data/alnizr_exp.csv',['Al','Ni','Zr'])


# #### all dev set ternaries for plotting #####
dev_plot_pred = load_prep_data('../data/triangles_glass_featurized.csv',allfeatures)
dev_plot_exp = load_prep_data('../data/hitp_glass_data_featurized.csv',allfeatures)

# # dev_plot_pred = clf.predict_proba(dev_plot_data)

# ###### Co Fe Zr ######
cofezr_list = ['Co-Fe-Zr','Co-Fe','Co-Zr','Fe-Zr','Co','Fe','Zr']

cofezr = dev_plot_pred[dev_plot_pred.ternary_name.isin(cofezr_list)]
cofezr_pred = clf.predict_proba(cofezr[features])
write_for_plotting(cofezr,cofezr_pred,'../data/cofezr_pred.csv',['Co','Fe','Zr'])

cofezr_real = dev_plot_exp[dev_plot_exp.ternary_name.isin(cofezr_list)]
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
fenbti_list = ['Fe-Nb-Ti','Fe-Nb','Fe-Ti','Nb-Ti','Fe','Nb','Ti']

fenbti = dev_plot_pred[dev_plot_pred.ternary_name.isin(fenbti_list)]
fenbti_pred = clf.predict_proba(fenbti[features])
write_for_plotting(fenbti,fenbti_pred,'../data/fenbti_pred.csv',['Fe','Nb','Ti'])

fenbti_real = dev_plot_exp[dev_plot_exp.ternary_name.isin(fenbti_list)]
# fenbti_real_val = real_class_to_proba(fenbti_real,fenbti_list)

# write_for_plotting(fenbti_real,fenbti_real_val,'../data/fenbti_exp.csv',['Fe','Nb','Ti'])



################
# Plotting CoFeZr at 2% Hitp training
x = .05
hitp_indices = hitp_train_data.index.tolist()
random.shuffle(hitp_indices)
hitp_len = len(hitp_indices)
x_ind = int(x*hitp_len)
inds = hitp_indices[0:x_ind]
hitp_train_data_n = hitp_train_data.iloc[inds]
all_train_data = pd.concat([train, hitp_train_data_n])

yall = all_train_data['Class'].astype(int)
clf = RandomForestClassifier(n_estimators=500,n_jobs=-1) #n_jobs: -1 runs on all avail cores
clf.fit(all_train_data[features],yall)

cofezr_pred = clf.predict_proba(cofezr[features])
write_for_plotting(cofezr,cofezr_pred,'../data/cofezr_pred_5percent.csv',['Co','Fe','Zr'])


# ################
# ################
# print("\nTesting HiTP learning curve:\n")




# # x = np.arange(numll)+1
# # x = x[::-1]
# # x = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
# x = [0,.025,.05,.075,.1,.2,.4,.6,.8,1]
# hitp_indices = hitp_train_data.index.tolist()
# random.shuffle(hitp_indices)
# hitp_len = len(hitp_indices)
# x_ind = [int(i*hitp_len) for i in x]

# numll = len(x)


# loglosses_test = np.zeros(numll)
# loglosses_train = np.zeros(numll)
# loglosses_alnizr = np.zeros(numll)
# loglosses_cofezr = np.zeros(numll)
# loglosses_fenbti = np.zeros(numll)
# acc_test = np.zeros(numll)
# acc_train = np.zeros(numll)
# acc_alnizr = np.zeros(numll)
# acc_cofezr = np.zeros(numll)
# acc_fenbti = np.zeros(numll)


# # hitp_train_data_n = hitp_train_data.iloc[x_ind[n]]

# for n in np.arange(len(x)):

#     if x_ind[n] == 0:
#         all_train_data = train
#     else:
#         # hitp_train_data_n = hitp_train_data.iloc[0::n,:]
#         inds = hitp_indices[0:x_ind[n]]
#         hitp_train_data_n = hitp_train_data.iloc[inds]

#         all_train_data = pd.concat([train, hitp_train_data_n])

#     yall = all_train_data['Class'].astype(int)
#     clf = RandomForestClassifier(n_estimators=500,n_jobs=-1) #n_jobs: -1 runs on all avail cores
#     clf.fit(all_train_data[features],yall)

#     test_pred = clf.predict(test[features])
#     test_proba = clf.predict_proba(test[features])
#     test_val = test['Class'].astype(int).values
#     logloss_test = metrics.log_loss(test_val,test_proba)
#     loglosses_test[n] = logloss_test
#     acc_test[n] = metrics.accuracy_score(test_val,test_pred)

#     train_val = train['Class'].astype(int).values
#     train_proba = clf.predict_proba(train[features])
#     logloss_train = metrics.log_loss(train_val,train_proba)
#     loglosses_train[n] = logloss_train

#     alnizr_proba = clf.predict_proba(alnizr_real[features])
#     alnizr_pred = clf.predict(alnizr_real[features])
#     alnizr_val = alnizr_real['Class'].astype(int).values
#     loglosses_alnizr[n] = metrics.log_loss(alnizr_val,alnizr_proba)
#     acc_alnizr[n] = metrics.accuracy_score(alnizr_val,alnizr_pred)

#     cofezr_proba = clf.predict_proba(cofezr_real[features])
#     cofezr_pred = clf.predict(cofezr_real[features])
#     cofezr_val = cofezr_real['Class'].astype(int).values
#     loglosses_cofezr[n] = metrics.log_loss(cofezr_val,cofezr_proba)
#     acc_cofezr[n] = metrics.accuracy_score(cofezr_val,cofezr_pred)

#     fenbti_proba = clf.predict_proba(fenbti_real[features])
#     fenbti_pred = clf.predict(fenbti_real[features])
#     fenbti_val = fenbti_real['Class'].astype(int).values
#     loglosses_fenbti[n] = metrics.log_loss(fenbti_val,fenbti_proba)
#     acc_fenbti[n] = metrics.accuracy_score(fenbti_val,fenbti_pred)

#     del clf


# plt.figure()
# plt.plot(x,loglosses_test,'r.-',label='Dev set')
# plt.plot(x,loglosses_alnizr,'b.-',label = 'AlNiZr')
# plt.plot(x,loglosses_cofezr,'g.-',label = 'CoFeZr')
# plt.plot(x,loglosses_fenbti,'m.-',label = 'FeNbTi')
# # plt.plot(1/x,loglosses_train,'b',label='train')
# plt.legend()
# plt.title('Log Loss: Dense Data Learning Curve')
# plt.ylabel('Log Loss')
# plt.xlabel('Fraction of Dense Training Data Included')
# plt.savefig('loglosses.pdf')
# plt.show()

# plt.figure()
# plt.plot(x,acc_test,'r.-',label='Dev set')
# plt.plot(x,acc_alnizr,'b.-',label = 'AlNiZr')
# plt.plot(x,acc_cofezr,'g.-',label = 'CoFeZr')
# plt.plot(x,acc_fenbti,'m.-',label = 'FeNbTi')
# # plt.plot(1/x,loglosses_train,'b',label='train')
# plt.legend()
# plt.title('Accuracy: Dense Data Learning Curve')
# plt.ylabel('Accuracy')
# plt.xlabel('Fraction of Dense Training Data Included')
# plt.savefig('acc.pdf')
# plt.show()

