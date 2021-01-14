import nltk
import warnings
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from string import punctuation
warnings.filterwarnings("ignore")

# nltk.download() # for downloading packages

import numpy as np
import random
import io

readFile=io.open('data.txt', 'r', errors = 'ignore')
data=readFile.read()
lowerData=data.lower() # converts to lowercase

#nltk.download('punkt') # first-time use only # for downloading packages
#nltk.download('wordnet') # first-time use only # for downloading packages

stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

sentences_tokens = nltk.sent_tokenize(lowerData) # converts to list of sentences 
word_tokens = nltk.word_tokenize(lowerData) # converts to list of words

def stem_words(tokens):
    return [stemmer.stem(token) for token in tokens]

def Normalize(text):
    text = text.lower().translate(str.maketrans('','', punctuation))
    text = stopword.remove(text)
    return stem_words(nltk.word_tokenize(text))

greetings = ['hey', 'hello', 'hi', 'hey dude']
bye = ['bye', 'bye-bye', 'goodbye', 'have a good day']
thank_you = ['thanks', 'thank you', 'terimakasih', 'thank you very much', 'terima kasih', 'thank you so much']
thank_response = ['sama-sama', 'senang bisa membantu', 'you\'re welcome.' , 'no problem.', 'glad to help.']

# Generating response
def response(user_response):
    chatterBox_response=''
    sentences_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=Normalize)
    tfidf = TfidfVec.fit_transform(sentences_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        chatterBox_response=chatterBox_response+"Maaf, kami tidak dapat memproses pertanyaan Anda."
        return chatterBox_response
    else:
        chatterBox_response = chatterBox_response+sentences_tokens[idx]
        return chatterBox_response

#Initialize user message
def bot_initialize(user_msg):
    if(user_msg == '/start'):
        bot_resp = """Hallo! Saya adalah Pelican dari Restoran 'The Diving Pelican', ada yang bisa saya bantu? \n(Format Booking Tempat)\n\n[Booking Tempat] \nNama :\nWaktu: (ex: Senin, 12-01-2021, 21.00) \n\nKetik 'Bye' untuk keluar.""" 
        return bot_resp
    elif('[booking tempat]' in user_msg.lower()):
        return 'Reservasi berhasil!'
    elif(user_msg.lower() == 'assalamualaikum'):
        return "Waalaikumsalam"
    
    user_msg = user_msg.translate(str.maketrans('','', punctuation))
    user_msg = stopword.remove(user_msg)
    user_response = stemmer.stem(user_msg.lower())

    flag = True
    while(flag==True):
        if(user_response not in bye):
            if(user_response in thank_you):
                bot_resp = random.choice(thank_response)
                return bot_resp
            elif(user_response in greetings):
                bot_resp = random.choice(greetings)
                return bot_resp
            else:
                bot_resp = response(user_response)
                sentences_tokens.remove(user_response)  # remove user question from sent_token that we added in sent_token in response() to find the Tf-Idf and cosine_similarity
                return bot_resp
        else:
            flag = False
            bot_resp = random.choice(bye)
            return bot_resp

