import sys
import pandas as pd
pd.options.display.max_columns = 30
import numpy as np
from time import time
#
import warnings 
warnings.filterwarnings('ignore')
#
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
#
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer#, HashingVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.utils import parallel_backend
#
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC#, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
#
from joblib import dump, load
# custom
from preprocessing import load_data, split_data, preprocessing, clean_text


# prepare models pipeline
def benchmark(frame, x, y, models, test_size=0.20, train_model=True, model_target='all', report=True):
    
    # 1.
    X_train, Y_train, X_test, Y_test, X_evalua, Y_evalua  = split_data(frame, x, y, test_size)
    
    # 2.
    pipeline = {}
    # iter
    for name, model in models.items():
        # specific model train/test
        if model_target not in [name,'all']:
            continue
            
        # Define a pipeline combining a text feature extractor with classifier
        pipeline[name] = Pipeline([
                ('vect', CountVectorizer(stop_words=stop_words,preprocessor=clean_text)),
                ('tfidf', TfidfVectorizer(preprocessor=str)),
                ('clf', model),
            ], verbose=1)
        
        print('... Processing')
        # train the model 
        with parallel_backend('threading'):
            if train_model:
                print('Init train {}'.format(name))
                pipeline[name].fit(X_train, Y_train)
                print('End train {}'.format(name))
        
        # save or load model
        if train_model:
            dump(pipeline[name], 'models/{}_{}.joblib'.format(y,name), compress=7 if name=='RFC' else 0) # compress 1 low 9 high, RFC is too big
        else:
            pipeline[name] = load('models/{}_{}.joblib'.format(y,name)) 
        print('Save/load model {}_{}'.format(y,name))
        
        # test the model 
        with parallel_backend('threading'):
            if not report: # print metrics...?
                continue
            pred = pipeline[name].predict(X_test)
            score = accuracy_score(Y_test, pred)
            print("accuracy_test:   %0.3f" % score)
            eval_ = pipeline[name].predict(X_evalua)
            score = accuracy_score(Y_evalua, eval_)
            print("accuracy_eval:   %0.3f" % score)
            
            #    
            print("classification report:")
            print('TEST\n',classification_report(Y_test, pred))
            print('EVAL\n',classification_report(Y_evalua, eval_))
            
            print("confusion matrix:")
            cm = confusion_matrix(Y_test, pred)
            print('TEST\n',cm)
            ConfusionMatrixDisplay(cm).plot()
            cm = confusion_matrix(Y_evalua, eval_)
            print('EVAL\n',cm)
            ConfusionMatrixDisplay(cm).plot()
            
    return pipeline

# train models
def train_codes(frame, x, y, models, test_size=0.20, train_model=True, model_target='all',report=True):
    
    # preprocessing
    df_code = (preprocessing(frame,x,y))
    
    # train benchmark
    benchmark(df_code,x,y,models, test_size=0.20, train_model=True, model_target='all',report=True)
    
    return
    
# call from main   
def run_train(path, x, y, test_size=0.20, train_model=True, model_target='all',report=True):
    df = load_data(path=path, extension="",output="data/corpus.csv")

    # define models
    models = {"MNB": MultinomialNB(fit_prior=True, class_prior=None),
          "SVC":LinearSVC(),
          "LogReg":LogisticRegression(solver='sag',n_jobs=-1),
          "XGB":XGBClassifier(n_jobs=-1,eval_metric='merror'),
          "RFC":RandomForestClassifier(n_jobs=-1),
          "KNN":KNeighborsClassifier(n_neighbors=10,n_jobs=-1)} # slow, use 10 neighbors

    # target
    train_codes(df, x, y, models, test_size=0.20, train_model=True, model_target='all',report=True)
    
