import regex as re
import numpy as np
import random
from tqdm import tqdm
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import sys
import pickle

## Class defination
class Neural_Model:
    
    def __init__(self,corpus):
        self.corpus = corpus
    
    def check_url(self, word):
        url_words = ["https:", "http:", "www.", ".co", ".org"]
        for i in url_words:
            if word.find(i)>=0:
                return True
        return False
    
    def tokens_per_word(self, word):
        if word in ["<HASHTAG>", "<MENTION>", "<URL>"]:
            return [word]
        word_len = len(word)
        s, e = word_len-1, 0
        for i in range(word_len):
            if (word[i].isalnum()) or word[i] in ["@", "#"]:
                s = i
                break
        for i in range(word_len)[::-1]:
            if word[i].isalnum():
                e = i
                break

        if s>=e:
            # entire string is non-alphanum
            return list(word)
        else:
            tokens = []
            # all elements till first alpha num
            for i in list(word)[:s]:
                tokens.append(i)
            # word
            tokens.append(word[s:e+1])
            #elements after word
            for i in list(word)[e+1:]:
                tokens.append(i)
        return tokens
    
    
    def Tokenizer(self, in_str):
        """
        Cleaning includes:
            -> All sentences have lower case.
            -> replace all \n with space
            -> replace multiple spaces with single space
            -> seperate punctuation from start/end of words
        Handle:
            -> Hashtags
            -> Mentions
            -> URLs
        """
        # Cleaning 
        in_str = in_str.lower()
        corpus = re.sub('\n',' ',in_str)
        corpus = re.sub(' +',' ',corpus)
        word_list = corpus.split()
        # seperate punctuations
        sep_punct = []
        for word in word_list:
            # generate tokens from everyword
            for i in self.tokens_per_word(word):
                sep_punct.append(i)

        # get tokens and pad with 3 <PAD>
        ### Look for mention, hastag, url
        tokens = ["<S>", "<S>", "<S>"]
        for i in sep_punct:
            if i[0]=="#":
                tokens.append("<HASHTAG>")
            elif i[0]=="@":
                tokens.append("<MENTION>")
            elif self.check_url(i):
                tokens.append("<URL>")
            else:
                tokens.append(i)
        tokens += ["</S>"]
        return tokens
      
    def set_unk(self,tokens_list):
        vocab = {}
        for tokens in tokens_list:
            for token in tokens:
                if token in vocab:
                    vocab[token] += 1
                else:
                    vocab[token] = 1
        thr = 1
        fin = []
        for tokens in tokens_list:
            fin.append([])
            for token in tokens:
                if vocab[token]<=thr:
                    fin[-1].append("<UNK>")
                else:
                    fin[-1].append(token)
        vocab = {}
        for tokens in fin:
            for token in tokens:
                if token in vocab:
                    vocab[token] += 1
                else:
                    vocab[token] = 1
        vocab = list(vocab.keys())
        vocab = { vocab[i]:i for i in range(len(vocab)) }
        self.vocab = vocab
        return fin
    
    def train_test_split(self,r):
        sents = self.corpus.split(".")
        l = len(sents)
        idx_test = random.sample(range(l),int(r*l))
        X_train, X_test = [], []
        for i in range(l):
            if i in idx_test:
                X_test.append(sents[i])
            else:
                X_train.append(sents[i])
        X_train = ".".join(X_train)
        X_test = ".".join(X_test)
        self.Xy_train, self.Xy_test = X_train, X_test
        return     
    
    def vocab_vec(self, word, label="train", purpose="train"):
        if word in self.vocab:
            num = self.vocab[word]
        else:
            num = self.vocab["<UNK>"]
        if label=="train":
            return num
        else:
            fin = np.zeros(len(self.vocab))
            fin[num] = 1
            return fin
    
    
    def gen_ngram(self,tokens_list, score=False):
        hist_l, w_l = [], []
        for tokens in tokens_list:
            for i in range(len(tokens))[3:]:
                hist, word = tokens[i-3:i], tokens[i]
                t = []
                t.append(self.vocab_vec(hist[0]))
                t.append(self.vocab_vec(hist[1]))
                t.append(self.vocab_vec(hist[2]))
                hist_l.append(t)
                if score==False:
                    w_l.append([self.vocab_vec(word, "test")])
                else:
                    w_l.append([self.vocab_vec(word)])
        return hist_l, w_l 
        
    def train(self):
        sents = self.Xy_train.split(".")
        tokens = [self.Tokenizer(sent) for sent in sents]
        # To account for unseen words, all words with freq <=thr are <UNK>
        tokens = self.set_unk(tokens)
        num_classes = len(self.vocab)
        # Generate n-grams
        X, y = self.gen_ngram(tokens)
        X_val, y_val = X[-50:], y[-50:]
        X, y = X[:-50], y[:-50]
        X, y = np.array(X).astype('int32').reshape((-1,3)), np.array(y).astype('int32').reshape((-1,num_classes))
        X_val, y_val = np.array(X_val).astype('int32').reshape((-1,3)), np.array(y_val).astype('int32').reshape((-1,num_classes))
        # Neural model stuff
        embedding_size = 2048
        model1 = Sequential()
        model1.add(Embedding(num_classes, embedding_size, input_length=3))
        model1.add(LSTM(1024))
        model1.add(Dense(800, activation = 'relu'))
        model1.add(Dense(num_classes))
        model1.compile(loss=BinaryCrossentropy(from_logits=True), optimizer = Adam(1e-4))
        model1.fit(X, y, batch_size=400, epochs=3, 
                       validation_data=(X_val, y_val))
        self.model = model1
        return
    
    def get_score(self, in_str):
        model = self.model
        sents = in_str.split(".")
        tokens = [self.Tokenizer(sent) for sent in sents]
        num_classes = len(self.vocab)
        # Generate n-grams
        X, y = self.gen_ngram(tokens, score=True)
        X = np.array(X).astype('int32').reshape((-1,3))
        prob = np.float64(1)
        y_pred = model.predict(X, verbose=0)
        for i in range(len(y_pred)):
            prob *= np.float64(y_pred[i][y[i]]/sum(y_pred[i]))
        pp_score = 1/(prob**(1/4))
        return pp_score
    
    def gen_file(self, file_name):
        ## gen file for train
        print("train set")
        fin_str, avg = "", 0
        sents = self.Xy_train
        sents = " ".join(sents.split("\n")).split(".")
        for sent in tqdm(sents):
            score = self.get_score(sent)
            fin_str += sent + "\t" + str(score) + "\n"
            avg += score
        avg /= len(sents)
        fin_str = str(avg) + "\n" + fin_str
        file_name = file_name.split("###")
        file_name = "train".join(file_name)                                    
        with open(file_name, "w") as text_file:
            text_file.write(fin_str)
        ## gen file for test
        print("test set")
        fin_str, avg = "", 0
        sents = self.Xy_test.split(".")
        for sent in tqdm(sents):
            score = self.get_score(sent)
            fin_str += sent + "\t" + str(score) + "\n"
            avg += score
        avg /= len(sents)
        fin_str = str(avg) + "\n" + fin_str
        file_name = file_name.split("train")
        file_name = "test".join(file_name)                                    
        with open(file_name, "w") as text_file:
            text_file.write(fin_str)
        return


path = sys.argv[1]
with open(path, 'rb') as file:
    loaded_nlm = pickle.load(file)

to_pred = input("input sentence: ")
score = loaded_nlm.get_score(to_pred)
print(score)