# Script to classify the low-throughput data with a random forest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn import preprocessing
import sys

######   Tuning parameters   ######
train_size = .8
######   Function definitiions   ######
# Function to separate ternaries by name
def ternaryName(row,elements):
    composition = row[elements].nonzero()[0]
    return '-'.join(sorted(elements[composition].values))

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
min_train = np.loadtxt('min_error_train.txt',dtype='str')
data['is_train'] = data['ternary_name'].isin(min_train)
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
#print(test_pred[test['ternary_name'].isin(['Al-Ni-Zr','Al-Ni','Al-Zr','Ni-Zr'])])
#print(test_val[test['ternary_name'].isin(['Al-Ni-Zr','Al-Ni','Al-Zr','Ni-Zr'])])
#test_proba = clf.predict_proba(test[features])
#print(test_proba[test['ternary_name'].isin(['Al-Ni-Zr','Al-Ni','Al-Zr','Ni-Zr'])])


######   Plot learning curve   ######
features = imp_feat
train_sizes = []
train_scores = []
test_scores = []
for subset in [.01,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]:
    clf = RandomForestClassifier(n_estimators=500,n_jobs=-1,oob_score=True)
    train_subset = train[np.random.uniform(0,1,len(train))<=subset]
    clf.fit(train_subset[features],train_subset['Class'].astype(int))
    train_sizes.append(len(train_subset))
    train_scores.append(clf.oob_score_)
    test_pred = clf.predict(test[features])
    test_scores.append(clf.score(test[features],test_val))
    del train_subset
    del clf
plt.plot(train_sizes,train_scores,'b-')
plt.plot(train_sizes,test_scores,'r-')
plt.xlabel('# of training samples')
plt.ylabel('OOB score')
plt.legend(['Training set','Test set'])
plt.savefig('RF_downsampling_linear')
plt.show()

plt.semilogx(train_sizes, test_scores,'r-')
plt.xlabel('# of training samples')
plt.ylabel('OOB score')
plt.legend(['Training set','Test set'])
plt.savefig('RF_downsampling_logx')
plt.show()
