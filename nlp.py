import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download("maxent_ne_chunker")
nltk.download("words")
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

text_file = open("milleplateaux.txt")
#text_file=open("difference.txt")
text = text_file.read()

words = word_tokenize(text, language="french")

stop_words = set(stopwords.words("french"))

filtered_list = []
diacritics = {',', '’', '.', '«', '»', ':', ')', '(', ';', '{', '}'}
stop_words.update(diacritics)
for word in words:
    if (word.casefold() not in stop_words):
        filtered_list.append(word)

fdist = FreqDist(filtered_list)

print(fdist.most_common(100))

def extract_ne(words):
    tags = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tags, binary=True)
    return set(
        " ".join(i[0] for i in t)
        for t in tree
        if hasattr(t, "label") and t.label() == "NE"
    )
frequenc_aut = extract_ne(words)
print(frequenc_aut)