import sys
import pandas as pd
pd.options.display.max_columns = 30
import numpy as np
#
import warnings 
warnings.filterwarnings('ignore')
#
from sklearn.model_selection import train_test_split
#
import glob
#
import spacy
nlp = spacy.load('en_core_web_md', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed


# Load data
def load_data(path,extension,output):
    files = glob.glob("{}/*/*{}".format(path,extension))
    li = []
    for filename in files:
        f = open(filename, "rb")
        text = str(f.read()[2:-1]) # to string and remove b''
        f.close()
        category_vs_id = filename.split("/")
        category = category_vs_id[-2] # solo me interesa el nombre de la folder de los files
        id_ = category_vs_id[-1].split(".")[0] # solo quiero el nombre sin la extension
        #print(text)
        li.append([id_,category,text]) # lo escribo en mi array
        
    df = pd.DataFrame(data=li, columns=['id','category','text'], dtype=str)
    df.info()
    #print(df.sample(5))
    df.to_csv(output,index=False,encoding='utf-8')
    return df
    
# clean dataset
def clean_df(frame, x, y): # frame=df, x=text, y=code
    # Fill empty and NaNs values with NaN
    clean = frame.fillna(np.nan)
    # remove null text
    clean = clean[~clean[x].isnull()].reset_index(drop=True)
    #print(clean[x].count())
    return clean
    
# Preprocessing
# text
def clean_text(text, ignore_url=True, letter_only=True, content_words=True, lemma=True):
    #print(text)
    text = str(text).replace("\n", " ") # replace \n with ""
    text = text.lower() # to lower 
    # add experimetn two
    doc = nlp(text)
    res = []
    for token in doc:
        t = token
        token = t.text
        pos = t.pos_
        if ignore_url and (token.startswith('https:') or token.startswith('http:')):
            continue
        if letter_only:
            if not token.isalpha():
                continue
            elif token.isdigit():
                token = '<num>'
        if content_words and pos not in ["NOUN","PROPN","ADV","ADJ","VERB"]: 
            continue
        if lemma: 
            token = t.lemma_
        # add    
        res += token,
        
    return ' '.join(res)

# create new df
def preprocessing(frame, x, y):
    # clean df
    df_code = (clean_df(frame,x,y))
    
    # clean target x
    #print(df_code[x].sample(1).tolist())
    df_code[x] = df_code[x].map(lambda dx: clean_text(dx))
    #print(df_code[[x,y]].sample())
    
    return df_code
    
# split train, test, evaluation
def split_data(frame, x, y, test_size=0.20):

    # split data
    train, test = train_test_split(frame, random_state=12122020, test_size=test_size, shuffle=True)
    
    # train x and y
    X_train = train[x]
    Y_train = train[y]
    # test x
    X_test = test[x]
    
    # reserve the half for eval (test)
    msk = np.random.rand(len(test)) <= 0.5
    # split test
    test_ = test[msk] # copy
    evalua = test[~msk]
    test = test_
    # test, evalua x and y
    X_test_ = X_test[msk]
    X_evalua = X_test[~msk]
    X_test = X_test_ # copy
    Y_evalua = evalua[y]
    Y_test = test[y]

    #print(X_train.shape,X_test.shape,X_evalua.shape)
    
    return X_train, Y_train, X_test, Y_test, X_evalua, Y_evalua  # 80% train, 10% test, 10% eval
