from typing import List

import streamlit as st
import pickle
import re
import nltk
import preprocessor as p
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time


def prediction(ds,tfidf_text):
    '''
    This function uses tfidf vector to predict the ticket-type.
    And returns the prediction as well asprediction probabilty too.
    '''

    # cyberbullying classification
    if ds == 1:
        # svm_tfidf
        svm_tfidf = pickle.load(open('svm_tf_1.pkl', 'rb'))
        tf_pred = svm_tfidf.predict(tfidf_text.toarray())
    elif ds==2:
        sgd_tfidf = pickle.load(open('sgd_tf.pkl', 'rb'))
        tf_pred = sgd_tfidf.predict(tfidf_text.toarray())

    with st.spinner(text='Predicting...'):
        time.sleep(3)

    # returning prediction with its probability
    return tf_pred


def preprocess_text(ds,text):
    '''
    This function allows to convert the text data into tf-idf vector and then returns it.
    '''
    with st.spinner(text='Analyzing tweet...'):
        time.sleep(3)

    # Replacing url,emoji,hashtags,mentions,smileys,numbers by the correspondng terms
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.HASHTAG, p.OPT.MENTION, p.OPT.SMILEY, p.OPT.NUMBER)
    pp_data = p.tokenize(text)

    # Lowercase
    lower_text = pp_data.lower()

    #Lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize_text(t):
            return [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize((t))]

    #Tokenization
    w_tokenizer = TweetTokenizer()

    #Removing punctuations
    def remove_punctuation(words):
        new_words= []
        for word in words:
                new_word = re.sub(r'[^\w\s]', '', (word))
                if new_word != '':
                    new_words.append(new_word)
        return new_words


    words = lemmatize_text(lower_text)
    words = remove_punctuation(words)

    #Stopword removal
    stop_words = set(stopwords.words('english'))
    clean_text = ' '.join(word.strip() for word in words if word not in stop_words)

    # checker
    if clean_text == '':
        st.error(
            'No content after preprocessing')
        st.stop()

    # transforming text
    transformer = TfidfTransformer()
    if ds == 1:
        loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open("tfidf_feature_1.pkl", "rb")))
    elif ds == 2:
        loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("tfidf_feature_2.pkl", "rb")))
    tfidf_feature = transformer.fit_transform(loaded_vec.fit_transform(np.array([clean_text])))

    #encode_vec = glove_en.encode(texts=[clean_text], pooling='reduce_mean')
    # returning tfidf text vector

    return tfidf_feature


def main():
    # app title
    #st.header('Twitter Cyberbullying Classification')

    html_temp = '''
    <div style="background-color:tomato; padding:20px; border-radius: 25px;">
    <h2 style="color:white; text-align:center; font-size: 30px;"><b>Twitter Cyberbullying Classification</b></h2>
    </div><br><br>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)
    # input
    text = st.text_area('Please enter tweet:')

    # predicting ticket_type
    if st.button('Predict'):

        # necessary requirements

        # no empty text
        if text.strip() == '':
            st.warning('No tweet is written! Kindly enter a tweet.')
            st.stop()

        # no punctuation
        #if str(re.sub('[^A-Za-z]+', ' ', text)).strip() == '':
        #   st.warning('You have written punctuations only. Kindly enter a correct tweet.')
        #    st.stop()

        #preprocessing of text
        tfidf_text = preprocess_text(1,text)

        # predicting ticket-type
        pred = prediction(1,tfidf_text)
        print(pred)
        # result display
        if pred == 1:
            value = 'Cyberbullying'
            result = 'The classification of the tweet is: ' + value
        else:
            tfidf_text_2 = preprocess_text(2, text)
            pred2 = prediction(2, tfidf_text_2)
            if pred2 == 2:
                value = 'None'
            elif pred2==3:
                value = 'Spam'
            elif pred2 == 4:
                value ='Sarcasm'
            elif pred2 == 5:
                value ='Profanity'
            result = 'The Non-cyberbullying text is classified as: ' + value
        print(value)
        st.success(result + '\n')


if __name__ == '__main__':
    main()
