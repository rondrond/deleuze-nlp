import nltk
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download("maxent_ne_chunker")
nltk.download("words")

text_file = open("machado.txt")
# text_file=open("difference.txt")
text = text_file.read()

words = word_tokenize(text, language="portuguese")

#stop_words = set(stopwords.words("portuguese"))

#words = [w for w in words if w.lower() not in stop_words]

fd = nltk.FreqDist(words)
new_text = nltk.Text(words)
concordance_list = new_text.concordance_list("igreja", lines=5, width=250)
print('List size: '+str(len(concordance_list)) + '\n')
sia = SentimentIntensityAnalyzer()
for sentence in concordance_list:
    print(sentence.line + ': \n')
    print(sia.polarity_scores(sentence.line))
    print('\n')
# print(sia.polarity_scores(concordance_list))

# filtered_list = []
# diacritics = {',', '’', '.', '«', '»', ':', ')', '(', ';', '{', '}'}
# stop_words.update(diacritics)
# for word in words:
#     if (word.casefold() not in stop_words):
#         filtered_list.append(word)

# fdist = FreqDist(filtered_list)

# print(fdist.most_common(100))


# def extract_ne(words):
#     tags = nltk.pos_tag(words)
#     tree = nltk.ne_chunk(tags, binary=True)
#     return set(
#         " ".join(i[0] for i in t)
#         for t in tree
#         if hasattr(t, "label") and t.label() == "NE"
#     )


# frequenc_aut = extract_ne(words)
# print(frequenc_aut)
