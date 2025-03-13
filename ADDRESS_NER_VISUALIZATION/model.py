import random
import pandas as pd
import numpy as np
import spacy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing  import FunctionTransformer
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

nlp=spacy.load("en_core_web_sm")

def transform(X):
    # Changing the shape of input for training
    a=np.array(X['Building_Number']).reshape(-1,1)
    b=np.concatenate([np.vstack(X[col]) for col in X if X[col].dtype==object],axis=1)
    return np.hstack([a,b])

def Word2Vec(X):
    # Converting strings to vectors
    X=pd.DataFrame(X)
    unnamed=True
    for idx,i in X.iterrows():
        for col,data in i.iteritems():
            if(type(data)!=float):
                i[col]=nlp(data).vector
            else:
                if unnamed==True:
                    X.rename(columns={col:'Building_Number'},inplace=True)
                    unnamed=False
        X.iloc[idx]=i
    return X

def train_model(X_train,y_train):
    # Pipeline for classification
    pipeline=Pipeline(
        steps=[
            ('Data Preprocessing',ColumnTransformer([
                ('Drop','drop',['Country','Address']),
                ("Filling null values for string columns",SimpleImputer(strategy="constant",fill_value=" "),['Building_Name', 'City', 'Recipient', 'Street_Name','Zip_Code', 'State']),
                ("Filling null values for numeric columns",SimpleImputer(strategy="constant",fill_value=0),['Building_Number']),
            ],remainder='passthrough')),
            ('String to vector',FunctionTransformer(Word2Vec)),
            ('Changing input shape',FunctionTransformer(transform)),
            ('Classifier',AdaBoostClassifier(random_state=10,n_estimators=1300))
        ]
    )
    return pipeline.fit(X_train,y_train)

def save_model(model):
    dump(model,'AdaBoost.joblib')

def load_model():
    return load('AdaBoost.joblib')

def main():

    # Loading data
    train_data=pd.read_csv("train.csv")
    test_data=pd.read_csv("test.csv")

    if displayFileCount:
        # Displaying number of records
        print()
        print("Total number of records in train.csv -",len(train_data))
        print("Total number of records in test.csv -",len(test_data))
        print("Total records -",len(train_data)+len(test_data))

    # Splitting data into train and validation
    train_data=train_data.sample(frac=1,random_state=10)

    val_data=train_data[100:]
    train_data=train_data[:100]

    # Separating inputs and labels
    X_train=train_data.drop(columns=['label'])
    y_train=train_data['label']

    X_val=val_data.drop(columns=['label'])
    y_val=val_data['label']

    X_test=test_data.drop(columns=['label'])
    y_test=test_data['label']

    # Getting model
    if train:
        print()
        print("Training model")
        ABC=train_model(X_train,y_train)
        save_model(ABC)
    else:
        print()
        try:
            ABC=load_model()
            print("Loaded model")
        except:
            raise Exception("No trained model available. Change the option 'train' to True.")

    # Getting predictions from model
    predictions_train=ABC.predict(X_train)
    predictions_val=ABC.predict(X_val)
    predictions_test=ABC.predict(X_test)

    if displayTPTN:
        print()
        # Computing total number of real addresses which are True
        print("Total number of real addresses which are True from train set -",sum((y_train==True)&(predictions_train==y_train)))
        print("Total number of real addresses which are True from val set -",sum((y_val==True)&(predictions_val==y_val)))
        print("Total number of real addresses which are True from test set -",sum((y_test==True)&(predictions_test==y_test)))

        # Computing total number of fake addresses which are False
        print("Total number of fake addresses which are False from train set -",sum((y_train==False)&(predictions_train==y_train)))
        print("Total number of fake addresses which are False from val set -",sum((y_val==False)&(predictions_val==y_val)))
        print("Total number of fake addresses which are False from test set -",sum((y_test==False)&(predictions_test==y_test)))
    
    if predictForRandomRecord:
        print()
        print("Prediction for a randomly selected record")
        # Selecting a random record and giving its prediciton
        idx=random.randint(0,len(test_data)-1)
        record=test_data.iloc[idx]
        x=record.drop(columns=['label'])
        y=record['label']
        print(pd.DataFrame({
            'Address':x['Address'],
            'Prediction':predictions_test[idx],
            'Actual':y
        },index=[0]))
    
    if displayRealFakeAddresses:
        print()
        # Displaying identified real and fake addresses
        real=pd.concat([train_data[predictions_train==True].Address,val_data[predictions_val==True].Address,test_data[predictions_test==True].Address])
        fake=pd.concat([train_data[predictions_train==False].Address,val_data[predictions_val==False].Address,test_data[predictions_test==False].Address])
        print("Real Addresses")
        print(real)
        print()
        print("Fake addresses")
        print(fake)

    if displayClassificationReport:
        preds=np.hstack([predictions_train,predictions_val,predictions_test])
        actuals=np.hstack([y_train,y_val,y_test])
        print()
        print("Confusion Matrix")
        cm=confusion_matrix(actuals,preds)
        # Displaying confusion matrix
        print(pd.DataFrame(cm,columns=['Predicted: NO','Predicted: YES'],index=['Actual: NO','Actual: YES']))
        print()
        TN=cm[0][0]
        FN=cm[1][0]
        FP=cm[0][1]
        TP=cm[1][1]
        # Displaying TP, FP, FN, TN
        print("True Positives(TP) -",TP)
        print("False Positives(FP) -",FP)
        print("False Negatives(FN) -",FN)
        print("True Negatives(TN) -",TN)
        print()
        # Displaying accuracy
        print("Accuracy Score -",accuracy_score(actuals,preds))

if __name__=="__main__":
    ######################################
    # Change options according to purpose
    ######################################
    train=True
    displayFileCount=True
    displayTPTN=True
    predictForRandomRecord=True
    displayRealFakeAddresses=True
    displayClassificationReport=True

    print()
    print("The model outputs")
    print("True (1) - Real Address")
    print("False (0) - Fake Address")
    main()