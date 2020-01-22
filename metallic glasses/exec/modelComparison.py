# Script to compare ML models for classifying low-throughput data
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn import preprocessing
import sys

######   Function definitiions   ######
# Function to separate ternaries by name
def ternaryName(row,elements):
    composition = row[elements].nonzero()[0]
    return '-'.join(sorted(elements[composition].values))
def write_for_plotting(mydata,pred,filename,elems):
    mydata['formula'] = ''
    for i,x in mydata.iterrows():
        tn = x['ternary_name']
        # elems = tn.split('-')^M
        # elems = ['Al','Ni','Zr']^M
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
    # return the real class values in the same form as clf.predict_proba^M
    # also can select specific ternaries and sub-ternaries^M
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

######   Lodd high-throughput data   ######
allfeatures = list(data)
hitp_data = load_prep_data('../data/hitp_glass_data_featurized.csv',allfeatures)
hitp_train_data = hitp_data[~hitp_data['ternary_name'].isin(['Co-Fe-Zr','Co-V-Zr','Fe-Nb-Ti'])]
hitp_train_data = hitp_train_data.reset_index()


######   Import training and test sets from min_error_*.txt files   #####
min_train = np.loadtxt('min_error_train.txt',dtype='str')
data['is_train'] = data['ternary_name'].isin(min_train)
train, test = data[data['is_train']==True], data[data['is_train']==False]
# add the (maybe downsampled) hitp data to the whole LB train data set
#train = pd.concat([train, hitp_train_data])
print('Number of observations in the training data:',len(train))
print('Number of observations in the test data:',len(test))
# Build classification vector (y)
y = train['Class'].astype(int)

allfeatures = list(data)
tri_alnizr = load_prep_data('../data/triangles_alnizr_featurized.csv',allfeatures)

######   Random Forest classifier   ######
sys.stdout.write('\nBuilding Random Forest classifier......... ')
# Create classifier
clf = RandomForestClassifier(n_estimators=500,n_jobs=-1) #n_jobs: -1 runs on all avail cores
# Train classifier on training data
clf.fit(train[features],y)
print('Done.')
# Determine feature importance
imp_feat = sorted(zip(clf.feature_importances_,train[features]))
imp_feat = [list(t) for t in zip(*imp_feat)]
imp_feat = imp_feat[1][-40:-1]
# Uncomment these lines to only use most important features
#features = imp_feat
#clf.fit(train[features],y)
# Predict classifications of test data
test_pred = clf.predict(train[features])
# Create vector with validation values of test data
test_val = test['Class'].astype(int).values
# Output the number of incorrect classifications
print('Random Forest classifier generated',np.sum(test_pred != test_val),
      'misclassifications out of',len(test_val),'resulting in',
      np.sum(test_pred != test_val)/len(test_val),'classification error.')
# Output the ROC curve and log-loss
test_proba = clf.predict_proba(test[features])
train_proba = clf.predict_proba(train[features])
train_proba1 = [x[0] for x in train_proba]
train_val = train['Class'].astype(int).values
test_proba1 = [x[0] for x in test_proba]
auroc = 1-roc_auc_score(test_val, test_proba1)
fpr, tpr, _ = metrics.roc_curve(test_val, test_proba1, pos_label=0)
fpr2, tpr2, _ = metrics.roc_curve(train_val, train_proba1, pos_label=0)
auroc2 = 1-roc_auc_score(train_val, train_proba1)
print('Random Forest classifier test set AUROC:',auroc)
print('Random Forest classifier training set AUROC:',auroc2)
print('Random Forest classifier test log-loss:',metrics.log_loss(test_val,test_proba))
print('Random Forest classifier train log-loss:',metrics.log_loss(train_val,train_proba))
print('Random Forest classifier F1 score:',metrics.f1_score(y,test_pred))

plt.figure()
plt.plot(fpr,tpr,color='darkorange',lw=2)

alnizr_pred = clf.predict_proba(tri_alnizr[features])
write_for_plotting(tri_alnizr,alnizr_pred,'../data/alnizr_pred_RF.csv',['Al','Ni','Zr'])

######   Logistic regression   ######
sys.stdout.write('\nBuilding Logistic Regression classifier......... ')
# Create classifier
clf = LogisticRegression()
# Train classifier on training data
clf.fit(train[features],y)
print('Done.')
# Predict classifications of test data
test_pred = clf.predict(train[features])
# Create vector with validation values of test data
test_val = test['Class'].astype(int).values
# Output the number of incorrect classifications
print('Logistic Regression classifier generated',np.sum(test_pred != test_val),
      'misclassifications out of',len(test_val),'resulting in',
      np.sum(test_pred != test_val)/len(test_val),'classification error.')
# Output the ROC curve and log-loss
test_proba = clf.predict_proba(test[features])
train_proba = clf.predict_proba(train[features])
train_proba1 = [x[0] for x in train_proba]
train_val = train['Class'].astype(int).values
test_proba1 = [x[0] for x in test_proba]
auroc = 1-roc_auc_score(test_val, test_proba1)
fpr, tpr, _ = metrics.roc_curve(test_val, test_proba1, pos_label=0)
fpr2, tpr2, _ = metrics.roc_curve(train_val, train_proba1, pos_label=0)
auroc2 = 1-roc_auc_score(train_val, train_proba1)
print('Logisitic Regression classifier test set AUROC:',auroc)
print('Logistic Regression classifier training set AUROC:',auroc2)

plt.plot(fpr,tpr,color='darkorange',lw=2)

print('Logistic Regression classifier test log-loss:',metrics.log_loss(test_val,test_proba))
print('Logistic Regression classifier train log-loss:',metrics.log_loss(train_val,train_proba))
print('Logistic Regression classifier F1 score:',metrics.f1_score(y,test_pred))

alnizr_pred = clf.predict_proba(tri_alnizr[features])
write_for_plotting(tri_alnizr,alnizr_pred,'../data/alnizr_pred_LG.csv',['Al','Ni','Zr'])

######   Support Vector Machine   ######
sys.stdout.write('\nBuilding Support Vector Machine classifier......... ')
# Create classifier
clf = SVC(probability=True)
# Train classifier on training data
clf.fit(train[features],y)
print('Done.')
# Predict classifications of test data
test_pred = clf.predict(train[features])
# Create vector with validation values of test data
test_val = test['Class'].astype(int).values
# Output the number of incorrect classifications
print('Support Vector Machine classifier generated',np.sum(test_pred != test_val),
      'misclassifications out of',len(test_val),'resulting in',
      np.sum(test_pred != test_val)/len(test_val),'classification error.')
# Output the ROC curve and log-loss
test_proba = clf.predict_proba(test[features])
train_proba = clf.predict_proba(train[features])
train_proba1 = [x[0] for x in train_proba]
train_val = train['Class'].astype(int).values
test_proba1 = [x[0] for x in test_proba]
auroc = 1-roc_auc_score(test_val, test_proba1)
fpr, tpr, _ = metrics.roc_curve(test_val, test_proba1, pos_label=0)
fpr2, tpr2, _ = metrics.roc_curve(train_val, train_proba1, pos_label=0)
auroc2 = 1-roc_auc_score(train_val, train_proba1)
print('Logistic Regression classifier test set AUROC:',auroc)
print('Logistic Regression classifier training set AUROC:',auroc2)

plt.plot(fpr,tpr,color='darkorange',lw=2)

print('Support Vector Machine classifier test log-loss:',metrics.log_loss(test_val,test_proba))
print('Support Vector Machine classifier train log-loss:',metrics.log_loss(train_val,train_proba))
print('Support Vector Machine classifier F1 score:',metrics.f1_score(y,test_pred))

alnizr_pred = clf.predict_proba(tri_alnizr[features])
write_for_plotting(tri_alnizr,alnizr_pred,'../data/alnizr_pred_SVM.csv',['Al','Ni','Zr'])

######   Neural Network   ######
sys.stdout.write('\nBuilding Neural Network classifier......... ')
# Create classifier
clf = MLPClassifier()
# Train classifier on training data
clf.fit(train[features],y)
print('Done.')
# Predict classifications of test data
test_pred = clf.predict(train[features])
# Create vector with validation values of test data
test_val = test['Class'].astype(int).values
# Output the number of incorrect classifications
print('Neural Network classifier generated',np.sum(test_pred != test_val),
      'misclassifications out of',len(test_val),'resulting in',
      np.sum(test_pred != test_val)/len(test_val),'classification error.')
# Output the ROC curve and log-loss
test_proba = clf.predict_proba(test[features])
train_proba = clf.predict_proba(train[features])
train_proba1 = [x[0] for x in train_proba]
train_val = train['Class'].astype(int).values
test_proba1 = [x[0] for x in test_proba]
auroc = 1-roc_auc_score(test_val, test_proba1)
fpr, tpr, _ = metrics.roc_curve(test_val, test_proba1, pos_label=0)
fpr2, tpr2, _ = metrics.roc_curve(train_val, train_proba1, pos_label=0)
auroc2 = 1-roc_auc_score(train_val, train_proba1)
print('Neural Network classifier test set AUROC:',auroc)
print('Neural Network classifier training set AUROC:',auroc2)

plt.plot(fpr,tpr,color='darkorange',lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.savefig('roc_compare.pdf')
plt.show()

print('Neural Network classifier test log-loss:',metrics.log_loss(test_val,test_proba))
print('Neural Network classifier train log-loss:',metrics.log_loss(train_val,train_proba))
print('Neural Network classifier F1 score:',metrics.f1_score(y,test_pred))

alnizr_pred = clf.predict_proba(tri_alnizr[features])
write_for_plotting(tri_alnizr,alnizr_pred,'../data/alnizr_pred_NN.csv',['Al','Ni','Zr'])
