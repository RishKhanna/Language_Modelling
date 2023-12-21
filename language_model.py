import regex as re
import sys

class Language_Model:
    
    def __init__(self, corpus, smoothing):
        self.corpus = corpus
        self.smoothing = smoothing
        self.hist_with_word = {}
        self.unigram = {}
    
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
            -> replace all \n with space
            -> replace multiple spaces with single space
            -> seperate punctuation from start/end of words
        Handle:
            -> Hashtags
            -> Mentions
            -> URLs
        """
        # Cleaning 
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
      
    def update_hist_word(self, sentences, n):
        if n==1:
            for sent in sentences:
                for word in sent:
                    if word in self.unigram:
                        self.unigram[word] +=1
                    else:
                        self.unigram[word] =1
        else:
            for sent in sentences:
                for i in range(len(sent)+1)[n:]:
                    hist, word = " ".join(sent[i-n:i-1]), sent[i-1]
                    # C[hist][word]
                    if hist in self.hist_with_word:
                        if word in self.hist_with_word[hist]:
                            self.hist_with_word[hist][word] += 1
                        else:
                            self.hist_with_word[hist][word] = 1
                    else:
                        self.hist_with_word[hist] = {word:1}
    
    def train(self):
        # Get all sentences from corpa
        sents = self.corpus.lower()
        sents = sents.split(".")
        sents = [ self.Tokenizer(sent + ".") for sent in sents]                    
        
        # 4,3,2,1-gram
        self.update_hist_word(sents,4)
        self.update_hist_word(sents,3)
        self.update_hist_word(sents,2)
        self.update_hist_word(sents,1)
        
    
    def P_MLE(self,hist,word):
        if len(hist)==0:
            if word in self.unigram:
                return self.unigram[word]/sum(self.unigram.values())
            else:
                return 0
        else:
            hist = " ".join(hist)
            if hist in self.hist_with_word:
                if word in self.hist_with_word[hist]:
                    return self.hist_with_word[hist][word]/sum(self.hist_with_word[hist].values())
                else:
                    return 0
            else:
                return 0
    
    def one_min_ladmba(self, hist, word):
        """ Part of Witten Bell Smoothing."""
        hist = " ".join(hist)
        num_types, num_times = 0,0
        if hist in self.hist_with_word:
            num_types = len(self.hist_with_word[hist])
            num_times = sum(self.hist_with_word[hist].values())
        if num_types==0:
            fin = 1
        else:
            fin = (num_types)/(num_types+num_times)
        return fin

    def P_WB(self,hist,word):
        lambda_ = 1 - self.one_min_ladmba(hist, word)
        if len(hist)==0:
            # https://www.ee.columbia.edu/~stanchen/e6884/labs/lab3/x207.html
            C_e, N1plus, V = sum(self.unigram.values()), len(self.unigram), len(self.unigram)
            fin = (C_e)/(C_e + N1plus)*self.P_MLE([],word) + (N1plus)/(C_e + N1plus)*(1/V)
        else:
            fin = lambda_*self.P_MLE(hist,word) + (1 - lambda_)*self.P_WB(hist[1:],word)
        return fin
    
    
    def T1(self, hist, word,d):
        fin = 1
        if word in self.hist_with_word[hist]:
            fin = max(self.hist_with_word[hist][word]-d, 0)
            fin /= sum(self.hist_with_word[hist].values())
        else:
            fin = 0
        return fin
    
    def lambda_KN(self,hist, d):
        fin = d * ( (len(self.hist_with_word[hist])) / (sum(self.hist_with_word[hist].values())) )
        return fin    
    
    def P_KN(self,hist,word):
        # https://en.wikipedia.org/wiki/Kneser-Ney_smoothing
        d = 0.2  # manually assigned constant
        if len(hist)==0:
            if word in self.unigram:
                fin = (self.unigram[word]-d)/sum(self.unigram.values())
                return fin
            else:
                # https://stats.stackexchange.com/questions/114863/in-kneser-ney-smoothing-how-are-unseen-words-handled
                return d/len(self.unigram)  # d/V
        else:
            hist_str = " ".join(hist)
            if hist_str in self.hist_with_word:
                # P_KN(h,w) = T1 + Lambda*P_KN(h[1:],w)
                fin = self.T1(hist_str, word, d) + self.lambda_KN(hist_str, d)*self.P_KN(hist[1:], word)
                return fin
            else:
                # Back-Off
                return self.P_KN(hist[1:],word)
    
    
    def get_score(self,in_str):
        in_str = in_str.lower()
        tokens = self.Tokenizer(in_str)
        # Find prob of sentence
        fin_score = 1
        if self.smoothing=="w":
            for i in range(len(tokens))[3:]:
                score = self.P_WB(tokens[i-3:i], tokens[i])
                fin_score *= score
        elif self.smoothing=="k":
            for i in range(len(tokens))[3:]:
                fin_score *= self.P_KN(tokens[i-3:i], tokens[i])
        # Find Perplexity from probablity
        # PP = 1/√p
        p = fin_score
        pp_score = (1/p)**0.5
        return pp_score


smooth,path  = sys.argv[1], sys.argv[2]
corpus = open(path, "rt").read()

lm = Language_Model(corpus, smooth)
lm.train()
in_str = input("input sentence: ")
score = lm.get_score(in_str)
print(score)