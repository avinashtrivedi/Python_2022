from gensim.models import KeyedVectors
import numpy as np
from gensim.models import Word2Vec
import pandas as pd
import warnings
import os
from gensim.parsing.preprocessing import preprocess_string
warnings.filterwarnings('ignore')

class Embedding:
    
    def __init__(self,filename):
        self.word2v_file = 'w2vec_10d.txt'
        self.filename = filename
    
    def preprocessing(self):
        
        df = pd.read_csv(self.filename,sep='\t')
        t = df[df['description'].isna() == False]
        t = t[t['kw_name']!='ToBeRemoved']
        data = t.copy()
        data.reset_index(drop=True,inplace=True)
        data['product_name'] = data['product_name'].apply(lambda x:x.strip())
        data['desc_and_productname'] = data['product_name'] + ' ' + data['description']
         
        all_description = []
        for description in data['desc_and_productname'].values:
            all_description.append(preprocess_string(description))
            
        return all_description
            
    def word2vec(self,list_of_string):
            
        
        if not os.path.exists(self.word2v_file):
            model = Word2Vec(list_of_string, 
                     min_count=1,   # word frequency
                     size=10,      # dimention of word embeddings
                     workers=8,     # Number of processors
                     sg=0, # sg=0 for cbow, sg=1 for skip gram
                     window=1,      # Context window
                    ) 
        
            # save
            model.wv.save_word2vec_format(self.word2v_file)
        w2v = KeyedVectors.load_word2vec_format(self.word2v_file)
        
        return w2v