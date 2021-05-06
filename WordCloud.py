import json as js
import string
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from hazm import *
from nltk.corpus import stopwords
from wordcloud_fa import WordCloudFa

f = open(input("Enter JSON File's address: "), encoding="utf-8")
data = js.load(f)
f.close()

i = data['messages']
text = ""

for j in range(len(i)):
    if type(i[j]['text']) is list:
        for x in range(len(i[j]['text'])):
            if type(i[j]['text'][x]) is str:
                text += (i[j]['text'][x]).strip() + " "
    elif str(i[j]['text']).strip() == "":
        continue
    else:
        text += (i[j]['text']) + " "

stop_words_main = []
with open("stopwords.txt", "r", encoding="utf-8") as f:
    Lines = f.readlines()
    for line in Lines:
        stop_words_main.append(line.strip("\n"))

text = Normalizer().normalize(text).strip().lower()
text = text.translate(str.maketrans('', '', string.punctuation))
text = text.translate(str.maketrans('', '', '’'))
text = text.translate(str.maketrans('', '', string.digits))
text = text.translate(str.maketrans('', '', "۰۱۲۳۴۵۶۷۸۹"))
text = text.translate(str.maketrans(' ', ' ', "\n"))

word_list = WordTokenizer().tokenize(text)
stop_words = stopwords.words('english')
punctuations = list(string.punctuation)
words = [word.strip() for word in word_list if
         word not in stop_words and word not in stop_words_main and word not in punctuations]

text = ""
for x in words:
    text += x + " "

# alice = np.array(Image.open("mask.png"))

word_cloud = WordCloudFa(persian_normalize=True, width=2000, height=2000, margin=20,
                         repeat=False, max_words=500)
frequencies = word_cloud.process_text(text)
wc = word_cloud.generate_from_frequencies(frequencies)
fig = plt.figure(figsize=(20, 20), facecolor=None)
plt.figure()
plt.imshow(word_cloud)
plt.axis('off')
plt.savefig('WordsCloud.png', dpi=2000, transparent=True)
plt.show()
