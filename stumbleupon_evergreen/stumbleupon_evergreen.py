#!/usr/bin/env python
"""
stumbleupon_evergreen.py
"""
import numpy as np
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from collections import defaultdict
from pymadlib.pyroc import *
import sys
import json
from copy import deepcopy
import scipy

from nltk.probability import FreqDist
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn import metrics

def bag_of_words_model(df, column_name, target='label', k=1000):
    """
    """
    pos_array = df[(df[target] == 1)][column_name].values
    neg_array = df[(df[target] == 0)][column_name].values

    pipeline = Pipeline([('tfidf', TfidfTransformer()),
                         ('chi2', SelectKBest(chi2, k=k)),
                         ('nb', MultinomialNB())])
    clf = SklearnClassifier(pipeline)

    pos = [FreqDist(word_list) for word_list in pos_array]
    neg = [FreqDist(word_list) for word_list in neg_array]

    add_label = lambda lst, lab: [(x, lab) for x in lst]

    trained_clf = clf.train(add_label(pos, 1) + add_label(neg, 0))

    return trained_clf


def bag_of_words_predictions(bow_model, all_array):
    """
    bow_model: trained bag of words model returned from bag_of_words_model()
    all_array: np.array[list[string]] | array of word lists for test examples
    """
    fd_array = [FreqDist(word_list) for word_list in all_array]

    pred = bow_model.batch_prob_classify(fd_array)

    scores = [p.prob(1) for p in pred]

    return scores


def alchemy_categories():
    ac_list = ['gaming'
              ,'recreation'
              ,'business'
              ,'religion'
              ,'computer_internet'
              ,'unknown'
              ,'culture_politics'
              ,'science_technology'
              ,'law_crime'
              ,'sports'
              ,'0'
              ,'weather'
              ,'health'
              ,'arts_entertainment']
    return ac_list


def preprocess(df):

    # process boilerplate
    d = {}
    title, body, url = [], [], []
    for i in range(len(df)):
        d = json.loads(df['boilerplate'][i])
        df['boilerplate'][i] = d
        try:    title.append(d['title'])
        except: title.append('')
        try:    body.append(d['body'])
        except: body.append('')
        try:    url.append(d['url'])
        except: url.append('')
    df['title'] = title
    df['body'] = body
    df['my_url'] = url

    # numwords features
    numwords_in_title, numwords_in_body, numwords_in_url = [], [], []
    for i in range(len(df)):
        try:    numwords_in_title.append(len(title[i].split()))
        except: numwords_in_title.append(0)
        try:    numwords_in_body.append(len(body[i].split()))
        except: numwords_in_body.append(0)
        try:    numwords_in_url.append(len(url[i].split()))
        except: numwords_in_url.append(0)
    df['numwords_in_title'] = numwords_in_title
    df['numwords_in_body'] = numwords_in_body
    df['numwords_in_my_url'] = numwords_in_url

    # length features
    len_title, len_body, len_url = [], [], []
    for i in range(len(df)):
        try:    len_title.append(len(title[i]))
        except: len_title.append(0)
        try:    len_body.append(len(body[i]))
        except: len_body.append(0)
        try:    len_url.append(len(url[i]))
        except: len_url.append(0)
    df['len_title'] = len_title
    df['len_body'] = len_body
    df['len_url'] = len_url

    # word list columns
    title_list = []
    for t in df['title'].values:
        try:    title_list.append(str(t).lower().split())
        except: title_list.append([''])
    body_list = []
    for b in df['body'].values:
        try:    body_list.append(str(b).lower().split())
        except: body_list.append([''])
    url_list = []
    for u in df['my_url'].values:
        try:    url_list.append(str(u).lower().split())
        except: url_list.append([''])
    df['title_list'] = title_list
    df['body_list'] = body_list
    df['url_list'] = url_list

    # alchemy category features
    for ac in alchemy_categories():
        df['is_'+ac] = np.array(df['alchemy_category'].values == np.array([ac]*len(df)), dtype=int)
        # print ac, df['is_'+ac][3], df['alchemy_category'].values[3]

    # raw content features
    len_html, numwords_in_html, html_list = [], [], []
    for i in range(len(df)):
        urlid = df['urlid'].values[i]
        f = open('data/raw_content/'+str(urlid), 'r')
        html = f.read().lower()
        f.close()

        len_html.append(len(html))
        numwords_in_html.append(len(html.split()))
        html_list.append(html.split())
    df['len_html'] = len_html
    df['numwords_in_html'] = numwords_in_html
    df['html_list'] = html_list

    return df


def to_remove(target='label'):
    x = ['url'
        ,'urlid'
        ,'boilerplate'
        ,'alchemy_category'
        ,'title'
        ,'body'
        ,'my_url'
        ,'title_list'
        ,'body_list'
        ,'url_list'
        ,'tfv_title'
        ,'tfv_body'
        ,'tfv_url'
        ,'tfv_data'
        ,'data_for_tfv'
        ,'html_list'
        ,target]

    return x


def split_df(df, split_frac=0.6):

    N = len(df)
    idx = range(N)
    shuffle(idx)
    split_idx = int(N*split_frac)

    df_train = df.iloc[idx[:split_idx]]
    df_test  = df.iloc[idx[split_idx:]]

    return df_train, df_test


def sub_models(df_train, df_rest, clf_info, bow_info, headers, target='label'):

    sub_model_dict = {}

    features_train = df_train[headers].values
    labels_train = df_train[target].values
    features_rest = df_rest[headers].values
    labels_rest = df_rest[target].values

    # fit clf models
    for clf_name, clf in clf_info.iteritems():
        print 'fitting '+clf_name+'...',
        clf_model = clf.fit(np.nan_to_num(features_train), labels_train)
        df_rest[clf_name+'_score'] = clf_model.predict_proba(np.nan_to_num(features_rest))[:, 1]
        sub_model_dict[clf_name] = clf_model
        print 'done'

    # fit bow models
    for bow_name, bow_dict in bow_info.iteritems():
        print 'fitting '+bow_name+'...',
        bow_model = bag_of_words_model(df_train
                                      ,column_name=bow_dict['column_name']
                                      ,target=target
                                      ,k=bow_dict['k'])
        df_rest[bow_name+'_score'] = bag_of_words_predictions(bow_model
                                                             ,df_rest[bow_dict['column_name']].values)
        sub_model_dict[bow_name] = bow_model
        print 'done'

    return df_rest, sub_model_dict


def main_dev():

    df = pd.read_csv('data/train_processed.tsv', sep='\t')
    df = preprocess(df)

    bow_info = {'bow_title':{'column_name':'title_list', 'k':8000}
               ,'bow_body':{'column_name':'body_list', 'k':25000}
               ,'bow_url':{'column_name':'url_list', 'k':10000}
               ,'bow_html':{'column_name':'html_list', 'k':25000}
               }

    df_train, df_rest = split_df(df, split_frac=0.7)

    rfc = RandomForestClassifier(n_estimators=1000
                                ,max_depth=10
                                ,min_samples_split=3
                                ,max_features='auto'
                                ,criterion='entropy'
                                ,n_jobs=8
                                ,compute_importances=True
                                ,oob_score=False)
    svm = SVC(probability=True)
#    lrc = LogisticRegression(penalty='l1', C=2, fit_intercept=False)
    lrc = LogisticRegression(penalty='l2', C=1, fit_intercept=True)

    headers = [h for h in df.columns if h not in to_remove(target='label')]
    clf_info = {'rfc':deepcopy(rfc)
               ,'lrc':deepcopy(lrc)
#               ,'svm':deepcopy(svm)
               }

    df_rest, sub_model_dict = sub_models(df_train, df_rest, clf_info, bow_info, headers=headers, target='label')

    # sub models finished --> scores are in df_rest

    df_train, df_test = split_df(df_rest, split_frac=0.5)

    clf = deepcopy(rfc)
#    clf = deepcopy(svm)
#    clf = deepcopy(lr)

    headers = [h for h in df_rest.columns if h not in to_remove(target='label')]

    features_train = np.nan_to_num(df_train[headers].values)
    labels_train = df_train['label'].values
    features_test = np.nan_to_num(df_test[headers].values)
    labels_test = df_test['label'].values

    clf_model = clf.fit(features_train, labels_train)

    scores = clf_model.predict_proba(features_test)[:, 1]

    # print out some results
    fi = sorted(zip(headers, clf_model.feature_importances_), key=lambda x: x[1])[::-1]
    for f, i in fi: print f, i
    fpr, tpr, thresholds = metrics.roc_curve(labels_test, scores, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    print '\nROC AUC:', auc


def main():

    df = pd.read_csv('data/train_processed.tsv', sep='\t')
    df = preprocess(df)

    bow_info = {'bow_title':{'column_name':'title_list', 'k':10000}
               ,'bow_body':{'column_name':'body_list', 'k':25000}
               ,'bow_url':{'column_name':'url_list', 'k':10000}
               ,'bow_html':{'column_name':'html_list', 'k':25000}
               }

    df_train, df_rest = split_df(df, split_frac=0.80)

    rfc = RandomForestClassifier(n_estimators=1000
                                ,max_depth=None
                                ,min_samples_split=3
                                ,max_features='auto'
                                ,criterion='entropy'
                                ,n_jobs=8
                                ,compute_importances=True
                                ,oob_score=False)
    svm = SVC(probability=True)
    lrc = LogisticRegression(penalty='l1', C=1, fit_intercept=True)

    headers = [h for h in df.columns if h not in to_remove(target='label')]
    clf_info = {'rfc':deepcopy(rfc)
               ,'lrc':deepcopy(lrc)
               }

    df_rest, sub_model_dict = sub_models(df_train, df_rest, clf_info, bow_info, headers=headers, target='label')

    # sub models finished --> scores are in df_rest

    clf = deepcopy(rfc)

    headers = [h for h in df_rest.columns if h not in to_remove(target='label')]
    features_train = np.nan_to_num(df_rest[headers].values)
    labels_train = df_rest['label'].values

    clf_model = clf.fit(features_train, labels_train)

    #===================
    # ALL MODELS FIT
    #===================

    df_TESTING = pd.read_csv('data/test_processed.tsv', sep='\t')
#    df_TESTING = pd.read_csv('data/train_processed.tsv', sep='\t')
    df_TESTING = preprocess(df_TESTING)
    urlid_TESTING = df_TESTING['urlid'].values

    headers_TESTING = [h for h in df_TESTING.columns if h not in to_remove(target='label')]
    features_TESTING = np.nan_to_num(df_TESTING[headers_TESTING].values)
#    labels_TESTING = df_TESTING['label'].values

    # get sub model scores
    sub_model_scores = {}
    for sub_model_name, sub_model in sub_model_dict.iteritems():
        if 'bow' in sub_model_name:
            bow_type = sub_model_name.split('_')[-1]+'_list'
            all_array = df_TESTING[bow_type].values
            scores = bag_of_words_predictions(sub_model, all_array)
        else:
            scores = sub_model.predict_proba(np.nan_to_num(features_TESTING))[:, 1]
        sub_model_scores[sub_model_name] = scores

    # now fit super model
    for sub_model_name, scores in sub_model_scores.iteritems():
        df_TESTING[sub_model_name] = scores
    headers_TESTING = [h for h in df_TESTING.columns if h not in to_remove(target='label')]
    features_TESTING = np.nan_to_num(df_TESTING[headers_TESTING].values)

    final_scores = clf_model.predict_proba(features_TESTING)[:, 1]

    # write scores to output
    out = open('results.csv', 'w')
    out.write('urlid,label\n')
    for i in range(len(final_scores)):
        out.write(str(urlid_TESTING[i])+','+str(final_scores[i])+'\n')
    out.close()

    # print out some results
    fi = sorted(zip(headers, clf_model.feature_importances_), key=lambda x: x[1])[::-1]
    for f, i in fi: print f, i

#    fpr, tpr, thresholds = metrics.roc_curve(labels_TESTING, final_scores, pos_label=1)
#    auc = metrics.auc(fpr,tpr)
#    print '\nROC AUC:', auc


if __name__ == '__main__':
#    main_dev()
    main()
