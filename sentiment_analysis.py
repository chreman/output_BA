# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import scipy.stats as stats
import config
import preprocessing as pp
import re

import statsmodels.api as sm

datafolder = config.datafolder


class ConfidenceIndexer(object):

    def __init__(self, inpath, testfile, outpath):

        self.outpath = outpath
        self.df = pd.read_pickle(datafolder+"verbrauchervertrauen_sentiment")
        self.df = self.df[self.df["sentiment"].apply(lambda x: x != [])] # filters for tweets with sentiment value
        self.texts = pd.read_pickle(datafolder+"verbrauchervertrauen_text")
        self.indicators = pd.read_table(datafolder+testfile, sep=",", index_col=0, parse_dates=True, header=0)
        self.searchframe = pd.merge(self.texts, self.df, "inner", on=["tweetid"])
        self.searchframe = self.searchframe.drop_duplicates("tweetid")
        self.get_positive_sentiment()
        self.set_date_index()
    

    def run_analysis(self):
        self.create_confidence_index()
        self.visualize_sentiment()
        self.count_texts()
        self.test_zipfs_law()
        self.user_activity_zipfs()


    def get_positive_sentiment(self):
        """
        Categorizes all tweets for positive or negative sentiment value.
        Creates a column "pos" with True / False boolean.
        """
        self.df["pos"] = self.df["sentiment"].apply(lambda x: np.mean(x) >= 0)


    def set_date_index(self):
        """
        Sets dataframe index to datetime.
        """
        self.df = self.df.set_index("datetime").copy()


    def create_confidence_index(self):
        """
        Creates a sentiment index as described in the paper.
        """

        df = self.df.copy()
        for column in df.columns:
            if not column in ["pos"]:
                df.drop(column, axis=1, inplace=True)
        self.conf = df.groupby("pos").resample("M", how="count").T
        self.conf.index = self.conf.index.droplevel(1)
        self.conf["total"] = self.conf.sum(axis=1)
        pos = self.conf[True]/self.conf["total"]
        neg = self.conf[False]/self.conf["total"]

        confidence = pos-neg
        confidence = confidence * 100
        confidence = confidence - confidence.mean()

        self.conf["confidence"] = confidence
        self.conf["mean"] = self.conf["confidence"].mean()


    def visualize_sentiment(self):
        """
        Plots the GfK-reference indicator and the twitter-confidence
        indicator on a time scale.
        """
        finalplot = plt.figure()
        plt.plot(self.conf.index, self.indicators["GfK"], label="GfK-Index")
        plt.plot(self.conf.index, self.conf["confidence"], label="Twitter-Index")
        plt.plot(self.conf.index, self.conf["mean"], label="Durchschnitt", linestyle="--")
        plt.legend(loc=0)
        plt.xlabel("Zeit")
        plt.ylabel("Indexwert")
        finalplot.set_size_inches(12.8, 9.6) # 1024x768 pixel
        finalplot.savefig(self.outpath+"sentiment.png")

        return finalplot


    def sentiment_around_word(self, word):
        """
        Performs a sentiment analysis on tweets filtered by keyword.
        """
        #self.searchframe["text"] = self.searchframe["text"].apply(lambda x: str(x).replace("ö", "oe"))
        wordframe = self.searchframe[self.searchframe["text"].apply(lambda x: word in x)] # leave only rows where word appears
        wordsentiment = wordframe[wordframe["sentiment"].apply(lambda x: x != [])]
        wordsentiment["pos"] = wordsentiment["sentiment"].apply(lambda x: np.mean(x) >= 0)
        wordindex = wordsentiment.set_index("datetime").copy()
        wordindex = wordindex.groupby("pos").resample("M", how="count").T
        wordindex.index = wordindex.index.droplevel(1)
        wordindex["total"] = wordindex.sum(axis=1)
        pos = wordindex[True]/wordindex["total"]
        neg = wordindex[False]/wordindex["total"]
        confidence = pos-neg
        confidence = confidence * 100
        wordindex["confidence"] = confidence

        wordsentimentfig = plt.figure()
        plt.plot(wordindex.index, wordindex["confidence"], label=word)
        plt.legend(loc=0)
        wordsentimentfig.set_size_inches(12.8, 9.6) # 1024x768 pixel
        wordsentimentfig.savefig(self.outpath+"wordsentimentfig_{0}.png".format(word))

        return wordsentimentfig


    def word_counter(self, word):
        """
        Plots tweet counts filtered by keyword.
        """
        word = self.searchframe[self.searchframe["text"].apply(lambda x: word in x)] # leave only rows where word appears
        for column in countframe.columns:
            if not column in ["datetime"]:
                countframe.drop(column, axis=1, inplace=True)
        wordindex = word.set_index("datetime").copy()
        wordindex = wordindex.resample("M", how="count")
        wordindex.index = wordindex.index.droplevel(1)

        wordcounterfig = plt.figure()
        plt.plot(wordindex.index, wordindex, label=word)
        plt.legend(loc=0)
        wordcounterfig.set_size_inches(12.8, 9.6) # 1024x768 pixel
        wordcounterfig.savefig(self.outpath+"wordcount_{0}.png".format(word))

        return wordcounterfig


    def get_granger(self, x,y, maxlag):
        ts = np.array([y,x]).T
        granger = sm.tsa.stattools.grangercausalitytests(ts, maxlag)
        granger = [(g[0], g[1][0]["params_ftest"]) for g in granger.items()]
        return granger


    def test_granger_causality(self, indicators):
        """
        Tests twitter-confidence indicator granger causality
        vs. all indicators given in the testfile.
        """

        #test = pd.read_table(datafolder+indicators, delimiter=",", parse_dates=True, index_col=0)
        #gfk = pd.read_table(datafolder+"consumerindex.csv", delimiter=",", parse_dates=True, index_col=0)
        #auftragsindex = pd.read_table(datafolder+"Auftragseingangsindex.csv", delimiter=",", parse_dates=True, index_col=0)

        for ind in self.indicators.columns:
            yield ind, self.get_granger(self.conf["confidence"], self.indicators[ind], 6)


    def test_zipfs_law(self):
        """
        Tests and plots whether Zipf's law applies to the test corpus.
        """

        tokens = self.texts["text"].apply(str)
        tokens = tokens.apply(pp.clean_text)
        tokens = tokens+" -99"
        tokens = tokens.apply(pp.tokenize)
        tokens = tokens.apply(pp.clean_urls)
        tokens = tokens.apply(pp.clean_twitter_handles)
        tokens = tokens.apply(pp.clean_tokens)

        import itertools
        from collections import Counter

        tokenslist = list(itertools.chain(*tokens))
        wordcounts = Counter(tokenslist)

        counts = [float(count) for count in wordcounts.values() if count > 20]
        ranks = [float(rank) for rank in stats.rankdata(counts, method="min")]

        ranks_log = [np.math.log10(rank) for rank in ranks]
        counts_log = [np.math.log10(count) for count in counts]
        rev = [-r+1 for r in ranks]
        minr = min(rev)
        rev = [r - minr for r in rev]

        zipfs_law = plt.figure()
        plt.plot(rev, counts, "o")
        plt.xscale("log")
        plt.yscale("log") 
        plt.xlabel("Rang (log)")
        plt.ylabel("Haeufigkeit (absolut, log)")
        zipfs_law.set_size_inches(12.8, 9.6) # 1024x768 pixel
        zipfs_law.savefig(self.outpath+"zipfs_law.png")

        return np.corrcoef(ranks, counts_log) # quick test of zipfs law


    def count_texts(self):
        """
        Plots tweet counts over time.
        """
        countframe = self.searchframe.copy()
        countframe["datetime"] = countframe["datetime"].astype(datetime.datetime)
        countframe = countframe.set_index("datetime")
        for column in countframe.columns:
            if not column in ["datetime", "tweetid"]:
                countframe.drop(column, axis=1, inplace=True)
        tweetcounts = countframe.groupby((countframe.index.year, countframe.index.month)).resample("M", "count")
        tweetcounts.index = tweetcounts.index.levels[2]
        
        tweetcountplot = plt.figure()
        plt.plot(tweetcounts.index, tweetcounts, label="Tweets")
        plt.legend(loc=0)
        plt.xlabel("Zeit")
        plt.ylabel("Anzahl Tweets pro Monat")
        tweetcountplot.set_size_inches(12.8, 9.6) # 1024x768 pixel
        tweetcountplot.savefig(self.outpath+"tweetcountplot.png")

        return tweetcountplot


    def user_activity_zipfs(self):
        """
        Plots Zipf's law or rather a power law-distribution
        of user activity.
        Tweets per user vs. number of users with this number of tweets.
        """
        userframe = self.searchframe.copy()
        for column in userframe.columns:
            if not column in ["datetime", "userid", "tweetid"]:
                userframe.drop(column, axis=1, inplace=True)
        userframe = userframe.set_index("datetime")
        userframe = userframe.groupby(userframe["userid"])
        usercounts = userframe.count(0)

        userranks = [float(rank) for rank in stats.rankdata(usercounts["tweetid"], method="average")]
        userranks_log = [np.math.log10(rank) for rank in userranks]
        urev = [-r+1 for r in userranks]
        minr = min(urev)
        urev = [r - minr for r in urev]

        useractivityplot = plt.figure()
        plt.plot(usercounts["userid"], urev, "o")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Tweets pro User insgesamt (log)")
        plt.ylabel("Anzahl User (log)")
        useractivityplot.set_size_inches(12.8, 9.6) # 1024x768 pixel
        useractivityplot.savefig(self.outpath+"useractivityplot.png")

        return useractivityplot
