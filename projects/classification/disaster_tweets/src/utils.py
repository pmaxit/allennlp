from typing import OrderedDict
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
# Function for converting emojis into word
from bs4 import BeautifulSoup

import string
from nltk.tokenize import TweetTokenizer
import re
import numpy as np
from wordcloud import STOPWORDS

tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

#clean data
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"couldnt" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"doesnt" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"havent" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"shouldnt" : "should not",
"that's" : "that is",
"thats" : "that is",
"there's" : "there is",
"theres" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"theyre":  "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not"}

puncts = puncts + list(string.punctuation)

def clean_text(x):
    x = str(x).replace("\n","")
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    x = re.sub('\d+', ' ', x)
    return x


def replace_typical_misspell(text):
    mispellings_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    def replace(match):
        return mispell_dict[match.group(0)]

    return mispellings_re.sub(replace, text)

def remove_space(string):
    string = BeautifulSoup(string,features='lxml').text.strip().lower()
    string = re.sub(r'((http)\S+)', 'http', string)
    string = re.sub(r'\s+', ' ', string)
    return string


def clean_data(df, columns: list):
    
    for col in columns:
        df[col] = df[col].apply(lambda x: remove_space(x).lower())        
        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))
        df[col] = df[col].apply(lambda x: clean_text(x))
        
    return df

#remove noise form text 
def remove_noise(text):
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r",", "", text)
    text = re.sub(r"\w+\d+", " numbers", text)
    text = re.sub(r'\d+','numbers', text)
    text = re.sub(r"\$", "dollar ", text)
    text = re.sub(r"\$+", "dollar ", text)
    text = re.sub(r"dollars", "dollar", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r":", "", text)
    text = re.sub(r" :", "", text)
    text = re.sub(r"\w+\-\w+", "", text)
    text = re.sub(r" -", "", text)
    text = re.sub(r" s ", "", text)
    text = re.sub(r" - ", "", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    #text = re.sub(r",", "", text)
    #text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    # text = re.sub(r"", "", text)
    #     #     remove urls
    # text = re.sub(r'http\S+', " ", text)
    # #     remove mentions
    # text = re.sub(r'@\w+',' ',text)
    # #     remove hastags
    # text = re.sub(r'#\w+', ' ', text)
    # #     remove digits
    # text = re.sub(r'\d+', ' ', text)
    # #     remove html tags
    # text = re.sub('r<.*?>',' ', text) 
    # text = text.replace(".", "")

    

    return text
    
        
#expanded_text = contractions_pattern.sub(expand_match, text)
#expanded_text = re.sub("'", "", expanded_text)
#return expanded_text
    
#remove stopwords

def remove_stopwords3(text):
    tokens = tokenizer.tokenize(text)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in STOPWORDS]
    
    return " ".join(filtered_words)
    
# def clean_text(df):

#     df["cleaned_text"] = df.text.map(lambda review:review.lower()).map(remove_noise).map(remove_noise2)
    
#     return df

def convert_emojis(text):
    if isinstance(text, list):
        res = []
        for t in text:
            token = t.text
            if token in EMOTICONS_EMO:
                res.append('_'.join(EMOTICONS_EMO[token].replace(',','').replace(':','').split()))
            res.append(token)
        return ' '.join(res)
    else:    
        for t in tokenizer.tokenize(text):
            if t in EMOTICONS_EMO:
                text = text.replace(str(t), '_'.join(EMOTICONS_EMO[t].replace(',','').replace(':','').split()))
        return text

def create_metafeatures(x):
    
    meta = OrderedDict({
        'word_count': len(x.split()),
        'unique_word_count': len(set(str(x).split())),
        'stop_word_count': len([w for w in str(x).lower().split() if w in STOPWORDS]),
        'url_count': len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]),
        'mean_word_length': np.mean([len(w) for w in str(x).split()]),
        'char_count': len(str(x)),
        'punctuation_count': len([c for c in str(x) if c in string.punctuation]),
        'hashtag_count': len([c for c in str(x) if c == '#']),
        'mention_count': len([c for c in str(x) if c == '@'])
    })
    metafeatures = list(meta.keys())
    return np.array([float(meta[f]) for f in metafeatures],dtype=np.float)


def text_preprocessing(text):
    
    #text = remove_noise(text)
    #text = remove_space(text).lower()

    text = replace_typical_misspell(text)
    text = clean_text(text)
    
    text = convert_emojis(text)
    return text
    
