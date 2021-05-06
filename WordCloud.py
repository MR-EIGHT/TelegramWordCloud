import json as js
import string

import matplotlib.pyplot as plt
from hazm import *
from wordcloud_fa import WordCloudFa

f = open(input("Enter JSON File's address: "), encoding="utf8")
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

stop_words = {':', '?', '؟', 'است', 'بود', 'شد', 'گشت', 'گردید', 'اما', 'ولی', '(', ')', 'از', 'به', 'با', 'و', 'که',
              'در', '(', ')', '،', 'یا', 'ای', 'هر', 'همه', 'هیچ', 'به هر', 'با هر', 'اگه', 'اگر',
              ':'}.union(set(stopwords_list()))

text = text.translate(str.maketrans('', '', string.punctuation))
text = text.lower()
text = text.translate(str.maketrans('', '', string.digits))
text = text.translate(str.maketrans('', '', "۰۱۲۳۴۵۶۷۸۹"))
text = Stemmer().stem(text)
text = Normalizer().normalize(text)

wordcloud = WordCloudFa(persian_normalize=True, width=1000, height=1000, margin=20, stopwords=stop_words, repeat=False)
frequencies = wordcloud.process_text(text)
wc = wordcloud.generate_from_frequencies(frequencies)

fig = plt.figure(figsize=(10, 10), facecolor=None)
plt.figure()
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('saved_figure.png', dpi=400, transparent=True)
plt.show()
