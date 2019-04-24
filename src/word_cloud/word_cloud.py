import numpy as np
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt

with open('../../data/word_dic.p', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    X_train, X_val, X_test, train_text, val_text, test_text, y_train, y_val, y_test, wordtoix, ixtoword = u.load()

text = train_text + val_text + test_text
label = y_train + y_val + y_test  # 0: normal, 1: depression, 2: PTSD, 3: Bipolar

s = ''
for sentence in text:
    s += sentence

text_normal, text_depression, text_PDSD, text_bipolar = '', '', '', ''

for i in range(len(label)):
    if label[i][0] == 1:
        text_normal += ' ' + text[i]
    elif label[i][1] == 1:
        text_depression += ' ' + text[i]
    elif label[i][2] == 1:
        text_PDSD += ' ' + text[i]
    else:
        text_bipolar += ' ' + text[i]

wordcloud_normal = WordCloud().generate(text_normal)
plt.imshow(wordcloud_normal, interpolation='bilinear')
plt.savefig('word_cloud_normal.png')
plt.clf()

wordcloud_depression = WordCloud().generate(text_depression)
plt.imshow(wordcloud_depression, interpolation='bilinear')
plt.savefig('word_cloud_depression.png')
plt.clf()

wordcloud_PDSD = WordCloud().generate(text_PDSD)
plt.imshow(wordcloud_PDSD, interpolation='bilinear')
plt.savefig('word_cloud_PDSD.png')
plt.clf()

wordcloud_bipolar = WordCloud().generate(text_bipolar)
plt.imshow(wordcloud_bipolar, interpolation='bilinear')
plt.savefig('word_cloud_bipolar.png')
plt.clf()
