""" 
This module gets called by the web app each time a site is requested. 

It orchestrates much of the user facing data processing including:
    * Checking for a valid web address
    * Spidering target site for article urls
    * Sending each article URL to a article scraping lambda to get text
    * Sending the accumulated text to the NLP lambda
    * Activating the plotting function to plot the NLP results

"""
import hashlib
import json
import os
from itertools import islice
from multiprocessing import dummy
from pprint import pprint
from time import sleep, time

import newspaper
import requests
import textblob
from unidecode import unidecode

from helpers import addDict, timeit
from plotter import plot

nlp_api = 'https://lbs45qdjea.execute-api.us-west-2.amazonaws.com/dev/dev_dnn_nlp'
scrape_api = 'https://x9wg9drtci.execute-api.us-west-2.amazonaws.com/prod/article_get'
analyzer = textblob.sentiments.PatternAnalyzer().analyze

test = False


class LambdaWhisperer:
    json_results = []

    def __init__(self):
        pass

    @timeit
    def scrape_api_endpoint(self, url_: str):
        response = json.loads(requests.put(scrape_api, data=url_).text)

        if 'message' in response:
            return None

        return {url_: response}

    @timeit
    def nlp_api_endpoint(self, url_text: dict):
        if not test:
            json.dump(url_text, open('./latest.json', 'w'))
            clean = url_text.values()
            response = json.loads(
                requests.put(nlp_api, data=unidecode(' ||~~|| '.join(url_text.values()))).text)
        else:
            json.dump(' ||~~|| '.join(url_text.values()), open('./latest.json', 'w'))
            print('saved results')

            exit()

        for r in sorted(response.items(), key=lambda kv: kv[1]):
            print(r)

        LambdaWhisperer.json_results = [response]

        return response

    @timeit
    def send(self, articles):

        return self.nlp_api_endpoint(articles)

    def snoop(self, cleaned):
        from collections import Counter
        c = Counter(cleaned.split(' '))
        print(c.most_common(15))


class Titles:
    collect = []


class GetSite:

    def __init__(self, url, name_clean=None, limit=100):  #50
        self.API = LambdaWhisperer()
        self.limit = limit
        self.url = self.https_test(url)
        if not name_clean:
            self.name_clean = self.strip()
        else:
            self.name_clean = name_clean
        self.hash = self.name_clean + '_' + hashlib.md5(str(time()).encode('utf-8')).hexdigest()[:5]

    def run(self):
        if not self.url:
            return False
        if self.url == 'ConnectionError':
            return self.url
        # Get list of newspaper.Article objs
        self.article_objs = self.get_newspaper()

        # Threadpool for getting articles

        self.article_objs = islice(self.article_objs, self.limit)
        self.articles = self.articles_gen()
        self.API.send(self.articles)

        if self.API.json_results:
            self.dump()
            self.save_plot()

        print(sorted(self.API.json_results[0].items(), key=lambda kv: kv[1], reverse=True))
        print(self.url)
        polarity, subjectivity = analyzer(self.articles)
        return self.num_articles, round(polarity, 3), round(subjectivity,
                                                            3), len(self.articles), self.hash

    def save_plot(self):
        plot(url=self.url, name_clean=self.hash)

    @timeit
    def articles_gen(self):

        url_list = [a.url for a in self.article_objs]

        res = {}
        third = self.limit // 3
        threadpool_1 = list(
            dummy.Pool(self.limit).imap_unordered(self.API.scrape_api_endpoint, url_list[:third]))

        threadpool_2 = list(
            dummy.Pool(self.limit).imap_unordered(self.API.scrape_api_endpoint, url_list[third:
                                                                                         third * 2]))
        sleep(.2)
        threadpool_3 = list(
            dummy.Pool(self.limit).imap_unordered(self.API.scrape_api_endpoint, url_list[third * 2:
                                                                                         third * 3]))
        results = threadpool_1 + threadpool_2 + threadpool_3
        for r in results:
            if r is not None:
                res.update(r)

        self.num_articles = len(res)

        return res

    def strip(self):
        return ''.join([
            c for c in self.url.replace('https://', '').replace('http://', '').replace('www.', '')
            if c.isalpha()
        ])

    def dump(self):

        j_path = './static/{}.json'.format(self.hash)
        with open(j_path, 'w') as fp:
            LambdaWhisperer.json_results[0].update({'n_words': len(self.articles)})

            json.dump(LambdaWhisperer.json_results, fp, sort_keys=True)

    @timeit
    def test_url(self, url_):
        print(url_)
        try:
            if requests.get(url_, timeout=(1, 5)).ok:
                print('connected to url'.format(url_))
                return url_
            else:
                print('failed to connect to url'.format(url_))
                return
        except requests.exceptions.ConnectionError:
            print('connection error')
            return 'ConnectionError'

    @timeit
    def https_test(self, url):
        if 'http://' or 'https://' not in url:
            return self.test_url('http://' + url) or self.test_url('https://' + url)
        else:
            return test_url(url)

    @timeit
    def get_newspaper(self):

        try:
            src = newspaper.build(
                self.url,
                fetch_images=False,
                request_timeout=2,
                limit=self.limit,
                memoize_articles=False)

        except Exception as e:
            print(e)
            print(self.url)
            return "No articles found!"
        print(len(src.articles))
        print(src.articles[0].url)

        return src.articles


if __name__ == '__main__':
    test = True

    @timeit
    def run(url, sample_articles=None):
        GetSite(url, sample_articles).run()

    run('naturalnews.com')
