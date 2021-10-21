import sys
import pandas as pd
pd.options.display.max_columns = 30
import numpy as np
from time import time
import glob
#
import warnings 
warnings.filterwarnings('ignore')
#
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report#, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.utils import parallel_backend
#
from joblib import dump, load
# custom
from preprocessing import split_data, preprocessing#, clean_text


# return top accuracy codes, i.e., i predict n codes for a text
def top_n_accuracy(probs,y_test,n): # revisar
    best_n = np.argsort(probs, axis=1)[:,-n:]
    y_true = np.array(y_test)
    return np.mean(np.array([1 if y_true[k] in best_n[k] else 0 for k in range(len(best_n))]))

# load train model    
def load_train_model(model):
    # load the model
    model = load(model)
    return model
 
# test a trained model
def predictions(frame, x, y, trained_model, test_size=0.20,):
    
    # 1.
    X_train, Y_train, X_test, Y_test, X_evalua, Y_evalua  = split_data(frame, x, y, test_size)
    
    # 2.
    pred = prob = []
    with parallel_backend('threading'):
            # predict
            pred = trained_model.predict(X_test)
            score = accuracy_score(Y_test, pred)
            print("accuracy:   %0.3f" % score)
            # proba
            try:
                prob = trained_model.predict_proba(X_test)
            except Exception as e:
                print(e,'try decision_function...')
                try:
                    prob = trained_model.decision_function(X_test)
                except Exception as e:
                    print(e,'prob None..')
                    
    return pred, prob, X_test, Y_test    

# call from main     
def run_predict(model, text, files=True):
    _cod_pred = None
    _confidence_pred = 0.0
    _2nd_code = None
    try:
        # prepare text
        f = open(text, "rb")
        _text = [str(f.read()[2:-1])] # to string and remove b''
        f.close()
        #_text = [clean_text(_text)]
        # load model
        _model = load_train_model(model)
        # predict
        _cod_pred = _model.predict(_text)[0]
    
        # prob
        try:
            prob_pred = _model.predict_proba(_text)
            print(prob_pred)
        except Exception as e:
            #print(e,'try decision_function...')
            try:
                prob_pred = _model.decision_function(_text)
            except Exception as e:
                #print(e,'prob None..')
                prob_pred = None
        _confidence_pred = prob_pred[0][np.argsort(prob_pred, axis=1)[:,-1:]][0][0] if prob_pred.any() else 0.0
    
        # second option
        _2nd_code = "".join(_model.classes_[[np.argsort(prob_pred, axis=1)[:,-2:][0][0]]])
    except Exception as e:
        print(e, 'Error running prediction... Please check the vars values.')
        
    return text, _cod_pred, float(_confidence_pred), _2nd_code
