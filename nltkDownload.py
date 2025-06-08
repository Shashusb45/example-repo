from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
print("Done")
