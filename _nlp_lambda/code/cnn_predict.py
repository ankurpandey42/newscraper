"""
This module calls the neural network on new data to make predictions.
It resides on AWS Lambda.

"""
import os
import pickle

import numpy as np
from tensorflow import keras

from add_dict import AddDict
from unidecode import unidecode
import pickle, json
Text = keras.preprocessing.text

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def preprocess_articles(article_list):

    def clean(self, seq):
        if len(seq):
            seq = unidecode(seq)
            return ' '.join(
                Text.text_to_word_sequence(
                    seq, filters='''1234567890!"#$%&()*+,-\n./—:;<=>?@[\\]^_`{|}~\t\'“”'''))

    def vectorize(text):
        lookup = json.load(open('lookup234.json'))
        for entry in text:
            entry = keras.preprocessing.text.text_to_word_sequence(entry)
            yield finalize([lookup[word] for word in entry if word in lookup])

    def finalize(entry):
        v_len = 1000

        if len(entry) >= v_len:
            print(len(entry))

            return np.array(entry[-v_len:]).reshape(1, -1)

        return np.array([0 for _ in range(v_len - len(entry))] + entry).reshape(1, -1)

    return list(vectorize(article_list))


labels = [
    'center', 'conspiracy', 'extreme left', 'extreme right', 'fake news', 'hate', 'high', 'left',
    'left-center', 'low', 'mixed', 'pro-science', 'propaganda', 'right', 'right-center', 'satire',
    'very high'
]

model = keras.models.load_model('CNN20k234.h5')


def predict(article):
    preds = model.predict(article)

    label_dict = {i: k for i, k in enumerate(labels)}

    pred_dict = {label_dict[i]: round(float(p), 6) for i, p in enumerate([x for x in preds.flatten()])}
    return pred_dict


def orchestrate(articles_concat):

    articles_clean = preprocess_articles(articles_concat.split('||~~||'))

    return [predict(chunk) for chunk in articles_clean]


'''
    # Add results
    results = AddDict()
    for r in [predict(chunk) for chunk in articles_clean]:
        results += r

    # Zero results

    # zero = predict(preprocess_articles(['the']))
    # for k, v in results.items():
    #     results[k] = (v / len(results)) - zero[k]
    return results
'''

if __name__ == '__main__':
    from pprint import pprint
    pprint(orchestrate('flouride alien dna  ||~~|| lizard people illuminati'))
