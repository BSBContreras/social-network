import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from TwitterUser import TwitterUser


    dataset = pd.read_csv('pre-processed.csv')

    dataset.drop(['Unnamed: 0'], axis=1, inplace=True)

    dataset.head()

    # separate labels and preprocessed_news from csv
    label = np.array(dataset["label"])
    preprocessed_news  = np.array(dataset["preprocessed_news"])

    cv = CountVectorizer()
    preprocessed_news = cv.fit_transform(preprocessed_news)

    #Train Naive Bayes Model
    #For multiple times of execution of our model, random state make sure that data values will be same for training and testing data sets.
    titletrain, titletest, labeltrain, labeltest = train_test_split(preprocessed_news, label, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(titletrain, labeltrain)

    headline = "Kane prevê dificuldades da Inglaterra contra a França: São os atuais campeões"

    data = cv.transform([headline]).toarray()
    #print(model.predict(data))

    print('Prestige Handler')
    print(model.predict(data))


    # https://github.com/roneysco/Fake.br-Corpus - Data CSV
    # based on this impementation https://github.com/Am1r8/Fake-news-Detector
