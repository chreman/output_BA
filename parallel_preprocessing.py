#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import subprocess

import string, re # for working with regular expressions and strings
from collections import Counter # data object for countable items
import pandas as pd
import numpy as np
from lxml import etree

import datetime

import utils.reformat
from IPython import parallel
from nltk.corpus import stopwords

import argparse

import config


opm = etree.parse(config.mainpath+"analyze/sentiment/OPdict.xml")


parser = argparse.ArgumentParser(description='Preprocessing workflow')
parser.add_argument('-f', '--file', required=True, dest='infile',
                    help='csv to preprocess')
parser.add_argument('--headers', required=False, dest='headers', nargs='*',
                    action='append', help='provide headers if not in csv')
parser.add_argument('-l', '--lang', required=False, dest='lang',
                    help='language of input, default english')
parser.add_argument('-s', '--sample', required=False, dest='samplesize',
                    help='apply preprocessing to random sample of raw data')
parser.add_argument('-o', '--output', required=True, dest='outfile',
                    help='file to store preprocessed data')
parser.add_argument('--parallel', dest='cores',
                    help='number of cores to parallelize')
parser.add_argument('--parser', dest='model',
                    help='POS Tagger model')
args = parser.parse_args()


rc = parallel.Client()
rc.block = True
dview = rc.direct_view()

if args.lang:
    language = args.lang
else:
    language = 'english'

swords = stopwords.words(language)
swords.extend(["rt", "fuer", "the", "wuerd", "imm", \
                         "heit", "onlin", "alt", \
                         "mehr", "neu", "via", "wurd", "gerad", \
                         "veroeffentlicht", "ueber"])

if language == "german":
    from nltk.stem.snowball import GermanStemmer as GS
    stemmer = GS()
if language == "english":
    from nltk.stem.snowball import EnglishStemmer as ES
    stemmer = ES()


def main(args):

    if args.headers:
        names = args.headers
        header = None
    else:
        names = None
        header = 0
    if args.lang:
        language = args.lang
    else:
        language = 'english'
    with dview.sync_imports():
        import string
        import numpy
        from lxml import etree
        from collections import Counter

    print "Now reading input data."
    df = pd.read_table(args.infile, sep=";", header=header, names=names[0])
    if args.samplesize:
        rows = np.random.choice(df.index.values, int(args.samplesize))
        df = df.ix[rows]

    df = df.drop_duplicates()
    #df.drop('url', axis=1, inplace=True) # if you want to get rid of unnecessary columns

    if "timestamp" in df.columns:
        df["datetime"] = df["timestamp"].apply(utils.reformat.reformat_time)
        df["datetime"] = df["datetime"].astype(datetime.datetime)


    dview["stopwords"] = swords
    dview["language"] = language
    dview["pos_tag"] = pos_tag

    print "Cleaning texts."
    tokens = df["text"].apply(str)
    tokens = tokens.apply(clean_text)
    tokens = tokens+" -99" # -99 is code for missing text value
    print "Recoding from latin-1 to utf-8."
    tokens = tokens.apply(recode)
    print "Tokenizing."
    tokens = tokens.apply(tokenize)
    print "Removing urls."
    tokens = tokens.apply(clean_urls)
    print "Removing twitter handles."
    tokens = tokens.apply(clean_twitter_handles)

    #################################################
    
    print "Scattering to parallel engines."
    dview.scatter('text_parts', tokens)
    print "Performing parallel POS tagging."
    dview.execute('tagged_texts = pos_tag(text_parts)')
    print "Gathering results from parallel engines."
    df["pos_tokens"] = dview.gather('tagged_texts')

    #################################################

    print "Getting all nouns."
    df["nouns"] = df["pos_tokens"].apply(get_nouns)
    df["nouns"] = df["nouns"].apply(clean_tokens)
    df["nouns"] = df["nouns"].apply(stem)
    
    token2id = create_token_dict(df["nouns"])
    print "Saving dictionary to file."
    with open(args.outfile+"_nouns.dict", 'w') as dictfile:
        cPickle.dump(token2id, dictfile)

    #################################################

    print "Getting all adjectives."
    df["adjectives"] = df["pos_tokens"].apply(get_adjectives)
    df["adjectives"] = df["adjectives"].apply(clean_tokens)
    df["adjectives"] = df["adjectives"].apply(stem)
    
    token2id = create_token_dict(df["adjectives"])
    print "Saving dictionary to file."
    with open(args.outfile+"_adjectives.dict", 'w') as dictfile:
        cPickle.dump(token2id, dictfile)
    
    #################################################

    print "Getting all tokens."
    df["alltokens"] = tokens
    df["alltokens"] = df["alltokens"].apply(clean_tokens)
    df["alltokens"] = df["alltokens"].apply(stem)

    token2id = create_token_dict(df["alltokens"])
    print "Saving dictionary to file."
    with open(args.outfile+"_alltokens.dict", 'w') as dictfile:
        cPickle.dump(token2id, dictfile)

    #################################################

    print "Getting sentiment."
    dview.execute("opm = etree.parse('{0}analyze/sentiment/OPdict.xml')".format(config.mainpath)) 
    dview.execute("root = opm.getroot()")
    dview["get_german_sentiment"] = get_german_sentiment
    df["sentiment"] = dview.map(analyze_german_sentiment, df["alltokens"])

    #################################################

    # remove unnecessary columns
    df.drop("userid", axis=1, inplace=True)
    df.drop("text", axis=1, inplace=True)
    df.drop("timestamp", axis=1, inplace=True)
    df.drop("pos_tokens", axis=1, inplace=True)

    # store preprocessing results to disj
    print "Saving dataframe to file."
    df.to_pickle(args.outfile)


    #################################################
    #################################################
    #################################################


def analyze_german_sentiment(tokens):
    tokens = [token for token in tokens if token]
    sentiment = [get_german_sentiment(token) for token in tokens]
    sentiment = [sent for sent in sentiment if sent]
    return sentiment


def get_german_sentiment(token):
    token = token.lower()
    query = "./*/term[text()='%s']" %token
    termnode = root.xpath(query)
    if termnode:
        sibs = [float(sib.attrib["polarity"]) for sib in termnode[0].itersiblings()]
        if sibs:
            return numpy.median(sibs)
        else:
            return None


def pos_tag(texts):

    from nltk.tag.stanford import POSTagger
    
    jar = config.mainpath+"analyze/SPOS/stanford-postagger.jar"
    if language == "german":
        model = config.mainpath+"analyze/SPOS/models/german-fast.tagger"
    if language == "english":
        model = config.mainpath+"analyze/SPOS/models/english-bidirectional-distsim.tagger"
    tagger = POSTagger(model, path_to_jar = jar, encoding="UTF-8")

    return tagger.tag_sents(texts)


def recode(text):
    text = text.decode("latin-1")
    return text.encode("utf-8")


def tokenize(text):
    """
    Splits a text into a list of tokens (unigrams).
    Returns a list of strings.
    
    >>> tokenize('Öffentlichkeit ist öffentlich')
    ['Öffentlichkeit', 'ist', 'öffentlich!']
    """
    tokens = text.split()
    return tokens


def clean_urls(tokens):
    return [token for token in tokens if not token.startswith("http") or token.startswith("www")]


def clean_twitter_handles(tokens):
    return [token for token in tokens if not token.startswith("@")]


def clean_text(text):

    text = text.replace("ä", "ae")
    text = text.replace("ö", "oe")
    text = text.replace("ü", "ue")
    text = text.replace("Ä", "Ae")
    text = text.replace("Ö", "Oe")
    text = text.replace("Ü", "Ue")
    text = re.sub(r'[^\x00-\x7F]+',' ', text)
    return text


def clean_tokens(tokens):
    """
    Takes a list of tokens as input, removes punctuation, replaces umlaute and lowers words.
    Returns a list of cleaned tokens.
    
    >>> clean_tokens(['Öffentlichkeit', 'ist', 'öffentlich!'])
    ['oeffentlichkeit', 'ist', 'oeffentlich']
    """
    delete_table = string.maketrans(string.ascii_lowercase,' ' * len(string.ascii_lowercase))

    tokens = [token.lower() for token in tokens]
    tokens = [str(token).translate(None, delete_table) for token in tokens]
    return tokens


def stem(tokens):
    """
    Takes a list of tokens and performs stemming.

    >>> stem(['oeffentlichkeit', 'ist', 'oeffentlich'], GermanStemmer)
    ['öffentlich', 'ist', 'öffentlich']
    """
    return [stemmer.stem(token) for token in tokens if stemmer.stem(token) not in swords]


def count_tokens(tokenlist):
    return Counter(tokenlist)


def get_all_tokens(texts):
    tokenlist = [token for tokens in texts for token in tokens]
    all_tokens = count_tokens(tokenlist)
    return all_tokens


def get_nouns(tokens):
    try:
        nouns = [token[0] for token in tokens if token[1] == "NN" and token[0] != "-99"]
    except:
        nouns = []
    return nouns


def get_adjectives(tokens):
    try:
        adjectives = [token[0] for token in tokens if token[1] == "ADJA" and token[0] != "-99"]
    except:
        adjectives = []
    return adjectives


def create_token_dict(texts):
    """
    Creates a dictionary of tokens from a dataframe.
    One maps tokens to ids, the other maps ids to tokens.
    """
    all_tokens = get_all_tokens(texts)
    unique_tokens = get_unique_tokens(texts)
    tokens_once = set([k for k, v in all_tokens.iteritems() if v == 1 ])
    tokens_more_than_once = unique_tokens - tokens_once

    print "Creating token dictionary."
    unique_tokens_id = list(enumerate(tokens_more_than_once))
    dict_token2id = {token[1]:token[0] for token in unique_tokens_id}
    dict_id2token = {token[0]:token[1] for token in unique_tokens_id}
    token2id = dict(dict_token2id.items() + dict_id2token.items())
    return token2id


def get_unique_tokens(texts):
    """
    Returns a set of unique tokens.
    
    >>> get_unique_tokens(['oeffentl', 'ist', 'oeffentl'])
    {'oeffentl', 'ist'}
    """
    unique_tokens = set()
    for text in texts:
        for token in text:
            unique_tokens.add(token)
    return unique_tokens



if __name__ == "__main__":
    main(args)
