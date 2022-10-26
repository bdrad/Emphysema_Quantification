#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import glob
import re
import pydicom
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, auc, roc_curve, confusion_matrix, roc_auc_score
import statsmodels.api as sm
import scipy
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import f1_score, confusion_matrix
from data_processing_utils import *
import warnings
warnings.filterwarnings("ignore")

basedir = ... # the directory where you saved all your metadata and emphysema score files
savedir = ... # the directory where you want to save metric data files


def combine_emphysema_cats(extent):
    if extent in ['none']:
        return 'none'
    elif extent in ['mild','mild to moderate', 'moderate']:
        return 'mild to moderate'
    elif extent in ['moderate to severe', 'severe']:
        return 'severe'
    else:
        return extent


def encode_emphysema_extent(extent):
    if extent == 'none':
        return 0
    elif extent == 'mild to moderate':
        return 1
    elif extent == 'severe':
        return 2
    else:
        return np.nan

def convert_df_to_lists(meta_df, score_col_indices):
    dfs = []
    non_score_col_indices = list(set(range(len(meta_df.columns)))-set(score_col_indices))
    for i in score_col_indices:
        df = meta_df.iloc[:,non_score_col_indices+[i]]
        df.rename(columns={df.columns.values[-1]:'emphysema score'},inplace=True)
        dfs.append(df)
    return dfs
    
def get_train_test_splits(df, seed = 0,
                          remove_outliers = True):
    '''
    5-fold train test split
    '''
    
    def not_outlier(summary_stats, emphy_score, doctor_label):
        LQ = summary_stats.loc[doctor_label, ('emphysema score', '25%')]
        UQ = summary_stats.loc[doctor_label, ('emphysema score', '75%')]
        IQR = UQ-LQ
        return emphy_score >= LQ-1.5*IQR and emphy_score <= UQ+1.5*IQR
    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True) 
    df = df.loc[~np.isnan(df['emphysema score']),:]
    X = df[['emphysema score', 'doctor note']].to_numpy()
    y = df['encoded extent (doctor)'].to_numpy()
    UCSF_stratified_generator = skf.split(X,y)
    
    train_test_pairs = []
    for train_index, test_index in UCSF_stratified_generator:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = pd.DataFrame(X_train[:,:2], columns = ['emphysema score', 'doctor note'])
        X_train['emphysema score'] = X_train['emphysema score'].astype('float')
        X_train['doctor note'] = X_train['doctor note'].astype('str')
        X_train['encoded extent (doctor)'] = y_train

        X_test = pd.DataFrame(X_test, columns = ['emphysema score', 'doctor note'])
        X_test['emphysema score'] = X_test['emphysema score'].astype('float')
        X_test['doctor note'] = X_test['doctor note'].astype('str')
        X_test['encoded extent (doctor)'] = y_test


        summary_stats = X_train.groupby('doctor note').describe()
    
        if remove_outliers:
            X_train['not outlier'] = X_train.apply(lambda x: not_outlier(summary_stats, x['emphysema score'], x['doctor note']), axis=1)
            select = (X_train['not outlier']) 
            #print('Number of training data:', len(X_train), 'Number of non-outliers:',np.sum(select))
        
        X_train = X_train.loc[select, :]
        
        train_test_pairs.append((X_train, X_test))
    
    return train_test_pairs


def train(transition_type, df):
    def one_repeat(i):
        X = df[['emphysema score']].copy().to_numpy()
        y = df['encoded extent (doctor)'].copy().to_numpy()
        
        
        if transition_type == ('none', 'mild to moderate'):
            encoding_cutoff = 0
        elif transition_type == ('mild to moderate', 'severe'):
            encoding_cutoff = 1

                
        boolean = y > encoding_cutoff
        y[boolean] = 1
        y[~boolean] = 0

        all_probs = []
        all_aucs = []
        all_cutoffs = []
        
        clf = LogisticRegression(random_state=i, C=1, class_weight='balanced')
        sample_num_0 = np.sum(y == 0)
        sample_num_1 = np.sum(y == 1)
        sample_weight_dict = {0:sample_num_0/(sample_num_0+sample_num_1),
                                 1:sample_num_1/(sample_num_0+sample_num_1)}
        sample_weight = np.array(list(map(lambda x:sample_weight_dict[x], y)))
        clf = clf.fit(X,y)
        k,b = clf.coef_[0][0], clf.intercept_[0]
        probs = clf.predict_proba(X)
        all_probs.append(probs)
        fpr, tpr, thresholds = roc_curve(y_score=probs[:,1], y_true=y)
        se = tpr
        sp = 1-fpr
        roc_auc = auc(fpr, tpr)
        all_aucs.append(roc_auc)
        idx = np.where(tpr-fpr==(tpr-fpr).max())[0][-1] #youden index
        thres = thresholds[idx]
        score_cutoff = (np.log((1-thres)/thres)-b)/k # convert from prob cutoff to score cutoff
        all_cutoffs.append(score_cutoff)
        return [fpr, tpr, thresholds, roc_auc], roc_auc, all_probs, all_cutoffs
    
    
    roc_params, roc_auc, all_probs, cutoffs = one_repeat(0)
 
    return roc_params, roc_auc, cutoffs

def get_cutoffs_and_aucs(df):
    
    roc_params1, aucs1, cutoffs1 = train(('none', 'mild to moderate'), df)
    cutoff1,_,auc1,_ =np.mean(cutoffs1), np.std(cutoffs1), np.mean(aucs1), np.std(aucs1)
    
    roc_params2, aucs2, cutoffs2 = train(('mild to moderate','severe'), df)
    cutoff2,_,auc2,_ =np.mean(cutoffs2), np.std(cutoffs2), np.mean(aucs2), np.std(aucs2)
    #print([aucs1, aucs2])
    return [[roc_params1, roc_params2],
            [cutoff1, cutoff2],
            [auc1, auc2],
           [aucs1, aucs2]]

def get_emphy_extent_new(score, cutoffs): # cutoffs is a length-2 vector
    if score < cutoffs[0]:
        return 'none'
    elif score < cutoffs[1]:
        return  'mild to moderate'
    else:
        return 'severe'
def pred_X_test(cutoffs, X_test):
    X_test = X_test.copy()
    X_test['reevaluated emphysema extent'] = X_test['emphysema score'].apply(lambda x:get_emphy_extent_new(x, cutoffs))
    X_test['encoded extent (re-predicted)'] = X_test['reevaluated emphysema extent'].map(encode_emphysema_extent).tolist()
    return X_test
def evaluate_X_test(X_test):
    diff = X_test['encoded extent (re-predicted)']-X_test['encoded extent (doctor)']
    diff1 = np.mean(abs(diff))
    categorical_pred = X_test.loc[:,'reevaluated emphysema extent']
    categorical_label = X_test.loc[:,'doctor note']
    labels = ['none', 'mild to moderate', 'severe']
    
    conf_mat = pd.DataFrame(confusion_matrix(categorical_label,categorical_pred,
                    labels=labels),
                      index = labels,
                      columns = labels)
    
    multiclass_acc = np.mean(categorical_pred==categorical_label)
    multiclass_f1_score = f1_score(categorical_pred, categorical_label, average='weighted')
    multiclass_kappa_score = kappa(categorical_pred, categorical_label)
    return X_test, conf_mat, [diff1, multiclass_acc,
                             multiclass_f1_score,
                             multiclass_kappa_score]


def get_stat_infos(dfs, repeats=10):
    
    stat_analysis_infos = {'no outlier':[{i:{} for i in range(len(dfs))} for _ in range(repeats)]}
    for k in stat_analysis_infos:
        for j in range(repeats):
            for i in range(len(dfs)):
                stat_analysis_infos[k][j][i]['roc params'] = []
                stat_analysis_infos[k][j][i]['aucs'] = []
                stat_analysis_infos[k][j][i]['cutoffs'] = []
                stat_analysis_infos[k][j][i]['all aucs'] = []
                stat_analysis_infos[k][j][i]['X_test'] = []

    for i in range(len(dfs)):
        for j in range(repeats):
            train_test_pairs = get_train_test_splits(dfs[i], seed=j)
            print(i,j)
            for X_train, X_test in train_test_pairs:
                roc_params, cutoffs, aucs, all_aucs = get_cutoffs_and_aucs(X_train)
                stat_analysis_infos['no outlier'][j][i]['roc params'].append(roc_params)
                stat_analysis_infos['no outlier'][j][i]['aucs'].append(aucs)
                stat_analysis_infos['no outlier'][j][i]['cutoffs'].append(cutoffs)
                stat_analysis_infos['no outlier'][j][i]['all aucs'].append(all_aucs)
                stat_analysis_infos['no outlier'][j][i]['X_test'].append(X_test)   

        
    for k in stat_analysis_infos:
        for j in range(repeats):
            for i in range(len(dfs)):
                stat_analysis_infos[k][j][i]['final cutoffs'] = np.mean(np.array(stat_analysis_infos[k][j][i]['cutoffs']),axis=0)
    return stat_analysis_infos

if __name__ == "__main__":

    original_df = pd.read_csv(basedir+'Reget_emphy_scores.csv',index_col=0) 
    # the file with extracted emphysema scores
    # my dataframe have 7 columns: accession number, and the emphysema scores from 6 different kernels
    original_df['accession number']=original_df['accession number'].astype('int')
    emphysema_df = pd.read_csv(basedir+'All_Patients_Extracted_Data.csv', index_col=0) # the file with emphysema extent
    emphysema_df.index = emphysema_df['Accession Number'].to_list()
    original_df['doctor note'] = emphysema_df.loc[original_df['accession number'].tolist(),'Emphysema Extent'].map(combine_emphysema_cats).tolist()
    original_df['encoded extent (doctor)'] = original_df['doctor note'].map(encode_emphysema_extent).tolist()
    filtered = original_df['doctor note']!='Not specified'
    training_df = original_df.loc[filtered,:]

    # train and get results stored in `stat_analysis_infos  `
    raw_dfs = convert_df_to_lists(training_df, score_col_indices = list(range(1,8)))
    stat_analysis_infos = get_stat_infos(raw_dfs)

    list_of_patches = ['3D ps=1', '3D ps=3', '3D ps=5', 
                    '2D ps=3', '2D ps=5', '2D ps=7']

    aucs= {i:np.vstack([np.array(stat_analysis_infos['no outlier'][j][i]['all aucs']) for j in range(10)]) for i in range(6)}
    # your final aucs

    AUC_col_names = [
                    'AUC mean (cat1)', 
                    'AUC SD (cat1)',
                    'AUC mean (cat2)', 
                    'AUC SD (cat2)']
    AUC_dict = {list_of_patches[i]:{AUC_col_names[j]:0 for j in range(4)} for i in range(7)}
    for j in range(2):
        for i in range(7):
            data = aucs[i][:,j]
            AUC_dict[list_of_patches[i]][AUC_col_names[j*2]] = float(np.mean(data))
            #print(AUC_dict)
            AUC_dict[list_of_patches[i]][AUC_col_names[j*2+1]] = float(np.std(data))
    pd.DataFrame(AUC_dict).round(3).to_csv(savedir+'AUC distributions.csv') # your AUC distributions  

    final_stats = {i:np.zeros((4)) for i in range(7)}
    final_stats_std = {i:np.zeros((4)) for i in range(7)}
    conf_mats = {i:[] for i in range(7)}

    for i in range(6):
        print(list_of_patches[i])
        final_stats_i = []
        for j in range(10):
            X_tests = []
            for k in range(5):
                X_test = stat_analysis_infos['no outlier'][j][i]['X_test'][k] 
                X_test = pred_X_test(stat_analysis_infos['no outlier'][j][i]['cutoffs'][k], X_test)
                X_tests.append(X_test)
            X_test = pd.concat(X_tests)
            X_test, conf_mat, stats = evaluate_X_test(X_test)
        
            label = X_test['encoded extent (doctor)'].astype('int').tolist()
            pred = X_test['encoded extent (re-predicted)'].astype('int').tolist()
            final_stats_i.append(np.array(stats))
            conf_mats[i].append(conf_mat)
        conf_mat = conf_mats[i][0]
        for j in range(1,10):
            conf_mat += conf_mats[i][j]
        conf_mats[i] = conf_mat / 10 # average of the confusion matrix of 10 train-test repeats
        final_stats_i = np.array(final_stats_i)
        final_stats[i] = np.mean(final_stats_i, axis=0)
        final_stats_std[i] = np.std(final_stats_i, axis=0)
        

    final_stats = pd.DataFrame(final_stats).round(3).rename(columns = {i:list_of_patches[i] for i in range(7)},
                                                index = {
                                                        0:'mean difference',
                                                        1:'multiclass accuracy',
                                                        2:'multiclass F score',
                                                        3:'multiclass kappa score'}) # your final stats
    final_stats_std = pd.DataFrame(final_stats_std).round(3).rename(columns = {i:list_of_patches[i] for i in range(7)},
                                                index = {
                                                        0:'mean difference',
                                                        1:'multiclass accuracy',
                                                        2:'multiclass F score',
                                                        3:'multiclass kappa score'}) # the standard deviation of your stats

    final_stats.to_csv(savedir+'final_stats.csv') 
    d = {list_of_patches[i]:conf_mats[i] for i in range(6)}
    all_conf_mats = pd.concat(d.values(), axis=1, keys=d.keys()) # your final confusion matrix