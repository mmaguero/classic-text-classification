## First install requirements:

```
python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt
```

### Then, download support files:

##### ...NLTK stopwords lists

`import nltk

nltk.download('stopwords')

exit()`

##### ...spaCy language model...

`python -m spacy download en_core_web_md`

## Now, execute...

### For train:

Two params: executable .py, and folder with data files

`python train.py "dataset"`

### For classify...

First param, executable .py, second, best model file, and variable arguments equals to files to classify:

`python classify.py "models/category_SVC.joblib" 'dataset/logistics/51131' 'dataset/weapons/54387' 'dataset/intelligence/178521'`

## When finished...

Close your venv

`deactivate`
