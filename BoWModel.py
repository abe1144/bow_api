import pandas as pd
import re
from sklearn.svm import LinearSVC
import pickle
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english')

class Model:
    def __init__(self):
        self.data = pd.read_csv('data_final.csv')

    def clean_text(self, text):
        text = re.sub(r'([^\s\w]|_)+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.lower()
        text = text.strip()
        return text
    
    #given an id, return the row in the dataframe
    # def get_row(self, id):
    #     data=self.data
    #     id = data[data['id'] == id]['id']
    #     id = data[data['id'] == id]['clean_text']


    #method that trains the model
    def train(self):
        print('training model...')
        data = pd.read_csv('data_final.csv')
        X = vectorizer.fit_transform(data['clean_text'].tolist())
        y = data['sentiment']
        svm = LinearSVC(random_state=42)
        svm.fit(X, y)
        #save model
        with open('svm_model.pk', 'wb') as f:
            pickle.dump(svm, f)

        #save vectorizer
        with open('vectorizer.pk', 'wb') as f:
            pickle.dump(vectorizer, f)

    def predict(self, text):
        #read in latest model
        with open('svm_model.pk', 'rb') as f:
            svm = pickle.load(f)
        
        #read in latest vectorizer
        with open('vectorizer.pk', 'rb') as f:
            vectorizer = pickle.load(f)

        normalized_text = self.clean_text(text)
        
        sentiment_prediction = svm.predict(vectorizer.transform([normalized_text]).toarray())[0]

        self.update_df(text, sentiment_prediction)

        record_id = int(self.data[self.data['clean_text'] == normalized_text]['id'])

        return record_id, text, sentiment_prediction

    
    def update_df(self, raw_text, label):
        #data = pd.read_csv('data_final.csv')
        normalized_text = self.clean_text(raw_text)

        #check if normalized_text already exist in datafile
        if self.data['clean_text'].str.contains(normalized_text).any() == False:

            id = len(self.data) + 1
            updated_data = self.data.append({'id':id,'clean_text':normalized_text, 'sentiment': label}, ignore_index=True)
            #overwrite existing data
            updated_data.to_csv('data_final.csv', index=False)
            #call to train
            self.data = updated_data
            self.train()
        else:
            print("entry already exists")
            pass
    
    def update_label(self, id, label):
        data = pd.read_csv('data_final.csv')
        data.loc[data.id == id, 'sentiment'] = label

        data.to_csv('data_final.csv', index=False)
        self.data = data