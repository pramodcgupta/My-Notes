#------------------------------------------------------------------------------------------
# 	 Pramodkumar Gupta
#	 NLP Code base
#	 Purpose: To keep all useful code handy
#------------------------------------------------------------------------------------------

import nltk

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

paragraph = """The Republic of India has two principal short names in both official and popular English usage, each of which is historically significant, India and Bharat. 
 The first article of the Constitution of India states that India, that is Bharat, shall be a union of states, implicitly codifying India and Bharat as equally official short names for the Republic of India. 
 A third name, Hindustan, is sometimes an alternative name for the region comprising most of the modern Indian states of the subcontinent when Indians speak among themselves. 
 The usage of Bharat, Hindustan, or India depends on the context and language of conversation. 
 the name for India in several Indian languages, is variously said to be derived from the name of either Dushyantas son Bharata or Rishabhas son Bharata. 
 At first the name Bharata referred only to the western part of the Gangetic Valley in North India, but was later more broadly applied to the Indian subcontinent and the region of Greater India, as was the name India.
 Today it refers to the contemporary Republic of India located therein. The name India is originally derived from the name of the river Sindhu (Indus River)  and has been in use in Greek since Herodotus (4th century BCE). 
 The term appeared in Old English as early the 9th century and reemerged in Modern English in the 17th century."""

# ----------------------------------------------------- Tokenization ----------------------------------------------------- 
# Tokenize sentences
sentences = nltk.sent_tokenize(paragraph)

# Tokenize words
words = nltk.word_tokenize(paragraph)


# ----------------------------------------------------- Stemming ----------------------------------------------------- 
from nltk.stem import PorterStemmer  
from nltk.corpus import stopwords

Stemmer = PorterStemmer()

for i in range(len(sentences)): 
    words=nltk.word_tokenize(sentences[i])
    words = [Stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i]= ' '.join(words)
	
""" Output:
Sentences: 
['the republ india two princip short name offici popular english usag , histor signific , india bharat .', 'the first articl constitut india state india , bharat , shall union state , implicitli codifi india bharat equal offici short name republ india .', 'A third name , hindustan , sometim altern name region compris modern indian state subcontin indian speak among .', 'the usag bharat , hindustan , india depend context languag convers .', 'name india sever indian languag , various said deriv name either dushyanta son bharata rishabha son bharata .', 'At first name bharata refer western part ganget valley north india , later broadli appli indian subcontin region greater india , name india .', 'today refer contemporari republ india locat therein .', 'the name india origin deriv name river sindhu ( indu river ) use greek sinc herodotu ( 4th centuri bce ) .', 'the term appear old english earli 9th centuri reemerg modern english 17th centuri .']
"""

# ----------------------------------------------------- Lemmatization ----------------------------------------------------- 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

Lemmatizer = WordNetLemmatizer()

for i in range(len(sentences)): 
    words=nltk.word_tokenize(sentences[i])
    words = [Lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i]= ' '.join(words)	

"""	Output:
Sentences: 
['The Republic India two principal short name official popular English usage , historically significant , India Bharat .', 'The first article Constitution India state India , Bharat , shall union state , implicitly codifying India Bharat equally official short name Republic India .', 'A third name , Hindustan , sometimes alternative name region comprising modern Indian state subcontinent Indians speak among .', 'The usage Bharat , Hindustan , India depends context language conversation .', 'name India several Indian language , variously said derived name either Dushyantas son Bharata Rishabhas son Bharata .', 'At first name Bharata referred western part Gangetic Valley North India , later broadly applied Indian subcontinent region Greater India , name India .', 'Today refers contemporary Republic India located therein .', 'The name India originally derived name river Sindhu ( Indus River ) use Greek since Herodotus ( 4th century BCE ) .', 'The term appeared Old English early 9th century reemerged Modern English 17th century .']	
"""

# ----------------------------------------------------- Bag of words/Document matrix ----------------------------------------------------- 

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

Lemmatizer = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)

corpus = []

for i in range(len(sentences)):
    review=re.sub('[^a-zA-Z]',' ',sentences[i])   # Remove all other charachets aprt from A-Z and a-z
    review=review.lower()    # Conver to lower case
    review=review.split()
    review = [Lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review= ' '.join(review)
    corpus.append(review)

# Create bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X=cv.fit_transform(corpus).toarray()   


# ----------------------------------------------------- TF-IDF Vectorization ----------------------------------------------------- 

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

Lemmatizer = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)

corpus = []

for i in range(len(sentences)):
    review=re.sub('[^a-zA-Z]',' ',sentences[i])   # Remove all other charachets aprt from A-Z and a-z
    review=review.lower()    # Conver to lower case
    review=review.split()
    review = [Lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review= ' '.join(review)
    corpus.append(review)

# Create TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X=cv.fit_transform(corpus).toarray()   




