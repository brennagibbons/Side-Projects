# from randomForest import *
# from randomForest import data,clf,features
import itertools
from sklearn import preprocessing
import pandas as pd


def write_for_plotting(data,pred,filename):
    data['formula'] = ''
    for i,x in data.iterrows():
        tn = x['ternary_name']
        # elems = tn.split('-')
        elems = ['Al','Ni','Zr']
        f = ''
        for e in elems:
            f = f+e
            f = f+str(x[e])
        # print f
        data.at[i,'formula'] = f


    with open(filename,'w') as writefile:
        for i,x in data.iterrows():
            p = pred[i][1]
            f = x['formula']
            elems = ["".join(x) for _, x in itertools.groupby(f, key=str.isalpha)]
            elemline = ' '.join(elems)
            writefile.write('{} {:.4f}\n'.format(elemline,p))

def load_prep_data(filename,allfeatures):
    ######   Load data to predict from ../data/comp_and_features.csv   ######
    data = pd.read_csv(filename,header=0)
    del data['gfa_measured'] #remove feature that is equivalent to class
    data['is_train'] = 0
    data['ternary_name'] = data.apply(lambda row: ternaryName(row,elements),axis=1)

    # populate elements that this data didn't have
    myFeatures = list(data)
    missing_features = [x for x in allfeatures if x not in myFeatures]

    for f in missing_features:
        data[f] = 0

    data[not_elements] = preprocessing.scale(data[not_elements])

    return data

# ######   Load data to predict from ../data/comp_and_features.csv   ######
# tri_alnizr = pd.read_csv('../data/triangles_alnizr_featurized.csv',header=0)
# del tri_alnizr['gfa_measured'] #remove feature that is equivalent to class
# tri_alnizr['is_train'] = 0
# tri_alnizr['ternary_name'] = tri_alnizr.apply(lambda row: ternaryName(row,elements),axis=1)

# # populate elements that this data didn't have
# allfeatures = list(data)

# alnizr_features = list(tri_alnizr)
# missing_features = [x for x in allfeatures if x not in alnizr_features]

# for f in missing_features:
#     tri_alnizr[f] = 0

# tri_alnizr[not_elements] = preprocessing.scale(tri_alnizr[not_elements])

# allfeatures = list(data)

# tri_alnizr = load_prep_data('../data/triangles_alnizr_featurized.csv',allfeatures)

# alnizr_pred = clf.predict_proba(tri_alnizr[features])

write_for_plotting(tri_alnizr,alnizr_pred,'alnizr_pred.csv')

alnizr_real = data[data.ternary_name.isin(['Al-Ni-Zr', 'Al-Ni','Ni-Zr','Al-Zr','Al','Ni','Zr'])]
alnizr_real_val = alnizr_real.as_matrix(columns=['Class'])
alnizr_real_val = list(alnizr_real_val.flatten())
alnizr_real_val = [alnizr_real_val,[-(x-1) for x in alnizr_real_val]]

# write_for_plotting(alnizr_real,alnizr_real_val,'alnizr_exp.csv')

# tri_alnizr['formula'] = ''
# for i,x in tri_alnizr.iterrows():
#     tn = x['ternary_name']
#     # elems = tn.split('-')
#     elems = ['Al','Ni','Zr']
#     f = ''
#     for e in elems:
#         f = f+e
#         f = f+str(x[e])
#     # print f
#     tri_alnizr.at[i,'formula'] = f


# with open('alnizr_pred.csv','w') as writefile:
#     for i,x in tri_alnizr.iterrows():
#         p = alnizr_pred[i][1]
#         f = x['formula']
#         elems = ["".join(x) for _, x in itertools.groupby(f, key=str.isalpha)]
#         elemline = ' '.join(elems)
#         writefile.write('{} {:.4f}\n'.format(elemline,p))

        

