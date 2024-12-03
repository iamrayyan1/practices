from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

text = "This is another example! Notice: it removes punctuation."
tokens = tokenizer.tokenize(text)
print(tokens)




# using regex
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

text = "This is a sample sentence, showing off the stop words filtration."

tokens = word_tokenize(text)
# Regular expression to match punctuation
cleaned_tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens if re.sub(r'[^\w\s]', '', token)]
print(cleaned_tokens)