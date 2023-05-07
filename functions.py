# import libraries
## infrastructure
import pandas as pd
import numpy as np
import math
import re

## machine learning
### sklearn for regression, clustering, LSA, evaluation
from sklearn.utils import shuffle
from sklearn import linear_model 
from sklearn.metrics import cohen_kappa_score,mean_absolute_error,mean_squared_error,accuracy_score,explained_variance_score,r2_score,confusion_matrix,ConfusionMatrixDisplay,classification_report,f1_score, silhouette_score,adjusted_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE,MDS
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans,SpectralClustering,AffinityPropagation,AgglomerativeClustering
### additional models
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import markov_clustering as mcl
import xgboost as xgb
import pingouin as pg
### graph support
import networkx as nx

## NLP tools
### Transformers, pytorch
from transformers import AutoModel, AutoTokenizer
import torch
from thinc.api import set_gpu_allocator, require_gpu
### spacy
import spacy
from spacy import displacy
from spacy.tokens import Span
from spacy.matcher import DependencyMatcher
from spacy.matcher import Matcher
from textacy import preprocessing
### additional tools
import nltk
from supar import Parser

## visualization tools
import matplotlib.pyplot as plt
import seaborn as sns





# define global constants
## All dependency tags
DepTagList = [i.strip() for i in '''acl, acomp, advcl, advmod, agent, amod, appos, attr, aux, auxpass, case, cc, ccomp, compound, conj, csubj, csubjpass, dative, dep, det, dobj, expl, intj, mark, meta, neg, nmod, npadvmod, nsubj, nsubjpass, nummod, oprd, parataxis, pcomp, pobj, poss, preconj, predet, prep, prt, punct, quantmod, relcl, xcomp'''.split(',')]

## Content word pos tags
CWPosTagList = ["NOUN","VERB","PRPON","INTJ","ADJ","ADV"]

## Function word pos tags
FWPoSTagList = ['ADP','AUX','CCONJ','DET','NUM','PART','PRON','SCONJ']
AllPosTagList = CWPosTagList + FWPoSTagList

## Phrases to look for
PhraseList = ['VP','NP','ADJP','PP','ADVP']

## Tense/aspect to look for
TenseList = ['Pres', 'Prog','Past','Perf']

## list for 143 measures
FeatureNameList = ['token/sent', 'verb/sent', 'noun/sent', 'adj/sent', 'adv/sent', 'verb/token', 'noun/token', 'adj/token', 'adv/token', 'content/function', 'verb_var', 'noun_var', 'adj_var', 'adv_var', 'tree_h/sent', 'vp/sent', 'np/sent', 'adjp/sent', 'advp/sent', 'pp/sent', 'acl/sent', 'acomp/sent', 'advcl/sent', 'advmod/sent', 'agent/sent', 'amod/sent', 'appos/sent', 'attr/sent', 'aux/sent', 'auxpass/sent', 'case/sent', 'cc/sent', 'ccomp/sent', 'compound/sent', 'conj/sent', 'csubj/sent', 'csubjpass/sent', 'dative/sent', 'dep/sent', 'det/sent', 'dobj/sent', 'expl/sent', 'intj/sent', 'mark/sent', 'meta/sent', 'neg/sent', 'nmod/sent', 'npadvmod/sent', 'nsubj/sent', 'nsubjpass/sent', 'nummod/sent', 'oprd/sent', 'parataxis/sent', 'pcomp/sent', 'pobj/sent', 'poss/sent', 'preconj/sent', 'predet/sent', 'prep/sent', 'prt/sent', 'punct/sent', 'quantmod/sent', 'relcl/sent', 'xcomp/sent', 'Pres/sent', 'Prog/sent', 'Past/sent', 'Perf/sent', 'NOUN', 'VERB', 'PRPON', 'INTJ', 'ADJ', 'ADV', 'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ', 'VP', 'NP', 'ADJP', 'PP', 'ADVP', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'predet', 'prep', 'prt', 'punct', 'quantmod', 'relcl', 'xcomp', 'Pres', 'Prog', 'Past', 'Perf', 'sent', 'token', 'content_word', 'function_word', 'unique_verb', 'unique_noun', 'unique_adj', 'unique_adv']



## patterns for rule matching
patterns = {
    ## copulars
    # I am happy
    'v_adj':[
        { # Auxiliary
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {'IN':["VERB",'AUX']}}
        },
        { # an adjective right child with adjectival complement tag
            "LEFT_ID": "verb",
            "REL_OP": ">++",
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {"POS": "ADJ", "DEP": "acomp"}
        }],
    # I am a student
    'v_noun(attr)':[
        { # Auxiliary or verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {'IN':["VERB",'AUX']}}
        },
        { #a right child with attribute tag
            "LEFT_ID": "verb",
            "REL_OP": ">++",
            "RIGHT_ID": "noun attr",
            "RIGHT_ATTRS": {"DEP": {'IN':["attr"],"NOT_IN":['expl']}}
        }],
    # I am in the house
    'v_prep':[
        { # Auxiliary
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {'IN':['AUX']}}
        },
        { # immediate preposition right childs
            "LEFT_ID": "verb",
            "REL_OP": ">+",
            "RIGHT_ID": "prep",
            "RIGHT_ATTRS": {"DEP": {'IN':["prep"]}},
        },
        { # a right child with object of preposition tag
            "LEFT_ID": "prep",
            "REL_OP": ">++",
            "RIGHT_ID": "pobj",
            "RIGHT_ATTRS": {"DEP": {'IN':["pobj"]}},
        }],
    # The mission is to kill him
    'v_to_do':[
        { # Auxiliary
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {'IN':['AUX']}}
        },
        { # right child with open clausal complement tag
            "LEFT_ID": "verb",
            "REL_OP": ">++",
            "RIGHT_ID": "xcomp",
            "RIGHT_ATTRS": {"DEP": {'IN':["xcomp"]},"POS":{'IN':['VERB','AUX']}},
        },
        { # ‘to’ as an immediate left child
            "LEFT_ID": "xcomp",
            "REL_OP": ">-",
            "RIGHT_ID": "to",
            "RIGHT_ATTRS": {"LOWER": {'IN':["to"]}},
        }],
    # The best thing is that he won. 
    "v_that":[
        { #Auxiliary
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {'IN':['AUX']}}
        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">++",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {"DEP": "ccomp"}
        },
        { 
            "LEFT_ID": "ccomp",
            "REL_OP": ">--",
            "RIGHT_ID": "that",
            "RIGHT_ATTRS": {"DEP": "mark"}
        }],
    
    ## transitive
    # I eat it
    'v_noun(dobj)':[ 
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": "VERB"}
        },
        { # right noun, proper noun or pronoun child with direct object tag
            "LEFT_ID": "verb",
            "REL_OP": ">++",
            "RIGHT_ID": "noun object",
            "RIGHT_ATTRS": {"POS": {'IN':["NOUN","PROPN","PRON"]}, "DEP": "dobj"}
        }],
    # I believe that it is wrong
    "v_that(objcl)":[
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {'IN':['VERB']}}
        },
        { # a right child with clausal complement tag
            "LEFT_ID": "verb",
            "REL_OP": ">++",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {"DEP": "ccomp"}
        },
        { # ‘that’ as a left child with marker tag
            "LEFT_ID": "ccomp",
            "REL_OP": ">--",
            "RIGHT_ID": "that",
            "RIGHT_ATTRS": {"DEP": "mark"}
        }],
    # I understand why you leave
    "v_wh(objcl)":[
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {'IN':["VERB"]}}
        },
        { # a right child with clausal complement tag
            "LEFT_ID": "verb",
            "REL_OP": ">++",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {"DEP": "ccomp","POS":{'IN':["VERB",'AUX']}}
        },
        { # ‘how’, ‘what’, ‘where’, ‘why’, ‘when’, or ‘who’ as a left child with direct object, attribute, or noun subject tag.
            "LEFT_ID": "ccomp",
            "REL_OP": ">--",
            "RIGHT_ID": "objclause",
            "RIGHT_ATTRS": {"DEP": {'IN':["dobj",'attr','nsubj']},"ORTH":{"IN":['how','what','where','why','when','who']}}
        }],
    
    ## v + to do/doing
    # Don't try to escape
    "v_to_v2":[
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": "VERB"}
        },
        { # a non-present-participle-verb right child with open clausal complement tag
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "xcomp",
            "RIGHT_ATTRS": {"DEP": "xcomp","TAG":{'NOT_IN':["VBG"]}}
        },
        { # to’ as a immediate left child
            "LEFT_ID": "xcomp",
            "REL_OP": ">-",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {"ORTH":'to'}
        }],
    # It derserves investigating. 
    "v_v2ing":[
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": "VERB"}
        },
        { # a present-participle verb right child with open clausal complement tag
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "xcomp",
            "RIGHT_ATTRS": {"DEP": "xcomp","TAG":'VBG'}
        }],
    
    ## transitive + dobj + oprd/(to) do/doing/done/adj
    # Please keep the room clean
    "v_noun(dobj)_oprd":[
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {"IN":["VERB",'AUX']}}
        },
        { # a child with direct object tag
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "obj",
            "RIGHT_ATTRS": {"DEP": "dobj"}
        },
        { # a child with object predicative tag
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "oprd",
            "RIGHT_ATTRS": {"DEP": "oprd"}
        }],
    # You can use it to drink
    "v_noun(subj)_to_v2":[
        { # verb (except ‘help’)
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {'IN':["VERB"]},'LEMMA':{'NOT_IN':["help"]}}
        },
        { # a non-present-participle and non-past-participle verb right child with clausal complement tag
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {"DEP": {'IN':["ccomp"]},"TAG":{"NOT_IN":['VBG','VBN']}}
        },
        { # ‘to’ as an immediate left child 
            "LEFT_ID": "ccomp",
            "REL_OP": ">-",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {"ORTH":'to'}
        },
        {  # a child with noun subject tag
            "LEFT_ID": "ccomp",
            "REL_OP": ">--",
            "RIGHT_ID": "nsubj",
            "RIGHT_ATTRS": {"DEP":'nsubj'}
        }],
    # It will make you cry.   
    "v_noun(subj)_v2":[
        { # verb (in ‘watch’, ‘see’, ‘notice’, ‘observe’, ‘make’, ‘let’, ‘have’, ‘look’, ‘listen’, ‘help’)
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {'IN':["VERB"]},"LEMMA":{'IN':["watch",'see','notice','observe','make','let','have','listen','look','help']}}
        },
        { # a non-present-participle and non-past-participle verb right child with clausal complement tag
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {"DEP": "ccomp","POS":{"IN":['VERB','AUX']},"TAG":{"NOT_IN":['VBG','VBN']}}
        },
        {  # a child with noun subject tag
            "LEFT_ID": "ccomp",
            "REL_OP": ">--",
            "RIGHT_ID": "nsubj",
            "RIGHT_ATTRS": {"DEP":'nsubj'}
        }],
    # I watched him jumping down
    "v_noun(subj)_v2ing":[
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {'IN':["VERB",'AUX']}}
        },
        { # a present-participle verb right child with clausal complement tag
            "LEFT_ID": "verb",
            "REL_OP": ">++",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {"DEP": "ccomp","TAG":'VBG'}
        },
        { # an immediate left child with noun subject tag
            "LEFT_ID": "ccomp",
            "REL_OP": ">-",
            "RIGHT_ID": "direct obj",
            "RIGHT_ATTRS": {"DEP": "nsubj"}
        }],
    # It makes our efforts wasted
    "v_noun(subj)_v2ed":[
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {'IN':["VERB",'AUX']}}
        },
        { # past-participle verb right child with clausal complement tag
            "LEFT_ID": "verb",
            "REL_OP": ">++",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {"DEP": "ccomp","TAG":'VBN'}
        },
        { #  an immediate left child with noun subject tag
            "LEFT_ID": "ccomp",
            "REL_OP": ">-",
            "RIGHT_ID": "direct obj",
            "RIGHT_ATTRS": {"DEP": "nsubj"}
        }],
    # It will make me miserable
    "v_noun(subj)_adj":[
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": "VERB"}
        },
        { # an adjective right child with clausal complement tag
            "LEFT_ID": "verb",
            "REL_OP": ">++",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {"DEP": "ccomp","POS":'ADJ'}
        },
        { # an immediate left child with noun subject tag
            "LEFT_ID": "ccomp",
            "REL_OP": ">-",
            "RIGHT_ID": "direct obj",
            "RIGHT_ATTRS": {"DEP": "nsubj"}
        }],
    # I want you to finish it
    "v_noun(dobj)_to_v2":[
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {'IN':["VERB"]}}
        },
        { # a right child with direct object tag
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "dobj",
            "RIGHT_ATTRS": {"DEP": "dobj"}
        },
        { # a child with open clausal complement tag
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "xcomp",
            "RIGHT_ATTRS": {"DEP":  {'IN':["xcomp"]}}
        },
        { #  ‘to’ has a left child
            "LEFT_ID": "xcomp",
            "REL_OP": ">--",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {"ORTH":'to'}
        }],
    
    ## di-transitive + dobj + dative/clause
    # I will give you some suggestions 
    "v_noun(dobj)_dative":[   
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {"IN":["VERB",'AUX']}}
        },
        { # a child with direct object tag
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "obj",
            "RIGHT_ATTRS": {"DEP": "dobj"}
        },
        { # a child with dative tag
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "dative",
            "RIGHT_ATTRS": {"DEP": "dative"}
        }],
    # I will tell you where it is
    "v_noun(dobj)_wh(cl)":[  
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {'IN':["VERB"]}}
        },
        { # a right child with direct object tag
            "LEFT_ID": "verb",
            "REL_OP": ">++",
            "RIGHT_ID": "dobj",
            "RIGHT_ATTRS": {"DEP": "dobj"}
        },
        { # a child with clausal complement tag
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "ccomp",
            "RIGHT_ATTRS": {"DEP":  {'IN':["ccomp"]}}
        },
        { # ‘that’, ‘how’, ‘what’, ‘where’, ‘why’, ‘when’, or ‘who’ as a left child
            "LEFT_ID": "ccomp",
            "REL_OP": ">--",
            "RIGHT_ID": "nsubj",
            "RIGHT_ATTRS": {"ORTH":{"IN":["that",'how','what','where','why','when','who']}}
        }],
    # I will tell you how to find it
    "v_noun(dobj)_wh(cl)_to_v2":[# help you do ...
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {'IN':["VERB"]}}
        },
        { # a right child with direct object tag
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "dobj",
            "RIGHT_ATTRS": {"DEP": "dobj"}
        },
        { # a child with open clausal complement tag
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "xcomp",
            "RIGHT_ATTRS": {"DEP":  {'IN':["xcomp"]}}
        },
        {  # ‘to’ as a left child
            "LEFT_ID": "xcomp",
            "REL_OP": ">--",
            "RIGHT_ID": "aux",
            "RIGHT_ATTRS": {"ORTH":'to'}
        },
        { # ‘that’, ‘how’, ‘what’, ‘where’, ‘why’, ‘when’, or ‘who’ as anther left child
            "LEFT_ID": "xcomp",
            "REL_OP": ">--",
            "RIGHT_ID": "pron",
            "RIGHT_ATTRS": {"ORTH":{"IN":["that",'how','what','where','why','when','who']}}
        },],
    
    ## verb in clauses
    # I believe that it is true
    "wh(objcl)_v_":[ 
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {'IN':["VERB",'AUX']}}
        },
        { # ‘that’, ‘how’, ‘what’, ‘where’, ‘why’, ‘when’, or ‘who’ as a left child with direct object or attribute tag
            "LEFT_ID": "verb",
            "REL_OP": ">--",
            "RIGHT_ID": "dobj",
            "RIGHT_ATTRS": {"DEP": {"IN":["dobj",'attr']},"ORTH":{"IN":["that",'how','what','where','why','when','who']}}
        },
        { # verb left parent
            "LEFT_ID": "verb",
            "REL_OP": "<--",
            "RIGHT_ID": "main verb",
            "RIGHT_ATTRS": {"POS": {'IN':["VERB",'AUX']}}
        }],
    # The time we spend together is precious
    "relcl_v_":[
        { # Verb with relative clause tag
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": "VERB","DEP": "relcl"}
        },
        { #  a left noun parent
            "LEFT_ID": "verb",
            "REL_OP": "<--",
            "RIGHT_ID": "relcl",
            "RIGHT_ATTRS": {"POS": "NOUN"}
        }],
    # That he was killed is unbelieveable
    "wh(csubj)_v_":[
        { # verb with clausal subject tag
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": "VERB","DEP": "csubj"}
        },
        { # ‘that’, ‘how’, ‘what’, ‘where’, ‘why’, ‘when’, or ‘who’ as a left child
            "LEFT_ID": "verb",
            "REL_OP": ">--",
            "RIGHT_ID": "wh word",
            "RIGHT_ATTRS": {"LOWER": {"IN":["that",'how','what','where','why','when','who']}}
        }
        ],
    
    ## Special sturctures
    # It is harmful to eat too much burger
    'It_is_?_to_v_':[
        { # verb with adverbial clause, clausal complement or open clausal complement tag
            "RIGHT_ID": "main verb", 
            "RIGHT_ATTRS": {"DEP": {"IN":['advcl','ccomp','xcomp']},"POS": {'IN':["VERB",'AUX']}}
        },
        { #  a left auxiliary parent
            "LEFT_ID": "main verb",
            "REL_OP": "<--",
            "RIGHT_ID": "be",
            "RIGHT_ATTRS": {"POS": {'IN':["VERB",'AUX']},"LEMMA":'be'}
        },
        { # ‘it’ as a left child.
            "LEFT_ID": "be",
            "REL_OP": ">--",
            "RIGHT_ID": "It subj",
            "RIGHT_ATTRS": {"ORTH": {'IN':["it","It","it'","It'"," It'"]}}
        },
        { # to’ as a left child
            "LEFT_ID": "main verb",
            "REL_OP": ">--",
            "RIGHT_ID": "to",
            "RIGHT_ATTRS": {"ORTH": {'IN':["to"]}}
        }],
    # He was told to step down
    "passive_v_":[
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {"IN":["VERB"]}}
        },
        { # a left child with passive auxiliary tag
            "LEFT_ID": "verb",
            "REL_OP": ">--",
            "RIGHT_ID": "auxpass",
            "RIGHT_ATTRS": {"DEP": "auxpass"}
        }],
    # Only in this way can we succeed
    "Only_prep_aux_v_inv":[
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {"IN":["VERB",'AUX']}}
        },
        { # a left child with auxiliary tag
            "LEFT_ID": "verb",
            "REL_OP": ">--",
            "RIGHT_ID": "left aux",
            "RIGHT_ATTRS": {"DEP": "aux"}
        },
        { # an immediate right sibling with noun subject tag
            "LEFT_ID": "left aux",
            "REL_OP": "$+",
            "RIGHT_ID": "subj",
            "RIGHT_ATTRS": {"DEP": "nsubj"}
        },
        { # left preposition child
            "LEFT_ID": "verb",
            "REL_OP": ">--",
            "RIGHT_ID": "prep",
            "RIGHT_ATTRS": {"DEP": "prep"}
        },
        { # ‘only’ as left child
            "LEFT_ID": "prep",
            "REL_OP": ">--",
            "RIGHT_ID": "only",
            "RIGHT_ATTRS": {"LOWER": "only"}
        }],
    # Not only can it help us
    "Only_aux_v_inv":[# help you do ...
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {"IN":["VERB",'AUX']}}
        },
        { # a left child with auxiliary tag
            "LEFT_ID": "verb",
            "REL_OP": ">--",
            "RIGHT_ID": "left aux",
            "RIGHT_ATTRS": {"DEP": "aux"}
        },
        { # an immediate right sibling with noun subject tag
            "LEFT_ID": "left aux",
            "REL_OP": "$+",
            "RIGHT_ID": "subj",
            "RIGHT_ATTRS": {"DEP": "nsubj"}
        },
        { # ‘only’ as a left child
            "LEFT_ID": "verb",
            "REL_OP": ">--",
            "RIGHT_ID": "only",
            "RIGHT_ATTRS": {"LOWER": "only"}
        }],
    # It is love that matters. 
    "It_is_?_that/who_v_":[# help you do ...
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": {"IN":["VERB"]}}
        },
        { # ‘that’ or ‘who’ as a left child
            "LEFT_ID": "verb",
            "REL_OP": ">--",
            "RIGHT_ID": "left subj",
            "RIGHT_ATTRS": {"ORTH":{"IN":['that','who']}}
        },
        { # is’ or ‘was’ as a left parent
            "LEFT_ID": "verb",
            "REL_OP": "<--",
            "RIGHT_ID": "be",
            "RIGHT_ATTRS": {"ORTH":{"IN":['is','was']}}
        },
        { # ‘it’ as a left child
            "LEFT_ID": "be",
            "REL_OP": ">--",
            "RIGHT_ID": "it",
            "RIGHT_ATTRS": {"LOWER": {"IN":['it',"it'"]}}
        }],
    # There is a dangerous bomb in the building.
    "expl_v_":[
        { # verb
            "RIGHT_ID": "verb", 
            "RIGHT_ATTRS": {"POS": "VERB"}
        },
        { # ‘it’ as a left child
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "expl",
            "RIGHT_ATTRS": {"DEP": "expl"}
        }],
}





# load models
## set cuda
set_gpu_allocator("pytorch")
require_gpu(0)

## load NLP models
NLP = spacy.load('en_core_web_trf')
SuPar = Parser.load('crf-con-en')

## mathcer 
matcher = DependencyMatcher(NLP.vocab)
for rule in patterns.keys():
    matcher.add(rule, [patterns[rule]])
RuleID = {NLP.vocab.strings[rule]:rule  for rule in patterns.keys()}





# Define functions 
## Find phrase with specified label
def FindPhrase(tree,phrase_label):
    count = 0
    flag = False
    if tree.label() == '_': # end of tree
            #print('end of tree')
            return 0
    
    children = [tree[j].label() for j in range(len(tree))]

    if tree.label() == phrase_label or (phrase_label == 'ADVP' and tree.label() in phrase_label): 
        if len(tree) >1 :
            count += 1 # found target phrase
        if not phrase_label in children : # do not search XP in XP, except XP does not have XP as its direct children, or XP has a SBAR child
            pass 
        elif 'SBAR' in children:
            if phrase_label in children :
                flag = True # continue searching, but do not search XP's direct child XP
        else:
            return count # stop searching
    for i in range(len(tree)):
        if flag and tree[i].label() == phrase_label:
            continue
        count += FindPhrase(tree[i],phrase_label)      
    return count



## Get subset combinations
def GetAllComb(n,NumList):
    # Get all subsets of length n in the given list, and in the original order
    res = []
    if n == 1: # [0,1,2] --> [[0],[1],[2]]
        return [[i] for i in NumList]
    if n == len(NumList): # [0,1,2] --> [0,1,2]
        return [[i for i in NumList]]
    if n >= 2: # for each element before len(NumList)-n, insert that element into position 0 for all subsets of length n-1 of the remaining elements
        for start in range(len(NumList)-n+1): 
            # (3,[0,1,2,3,4])
            last = GetAllComb(n-1,NumList[start+1:])
            # 0 + (2,[1,2,3,4]), 1 + (2,[2,3,4]), 2 + (2,[3,4])
            for comb in last:
                comb.insert(0,NumList[start])
            res += last
        return res
comb_dict = {i:{j+1:[]for j in range(i)}  for i in range(1,11)}
for i in range(1,11):
    for j in range(i):
        comb_dict[i][j+1] = GetAllComb(j+1,[c for c in range(i)])    
    


## Compute tree kernel
def KernelFunction(tree1,tree2,t1index,t2index,prev,tree1Index,tree2Index):
    # If already calculated, return result directly
    #print(t1index,t2index)
    if prev[t1index][t2index] != -1:
        return prev[t1index][t2index]
    # tree1 and tree2 are actually trees of token objects
    # get dep labels of all nodes, t1 and t2 are trees of dep labels
    t1 = (tree1[0].dep_,[i.dep_ for i in tree1[1] if i])
    t2 = (tree2[0].dep_,[i.dep_ for i in tree2[1] if i])
    # mu and lambda are decay factors, mu penalize tree height and lambda penalize tree length
    mu =0.9
    lambda_ = 0.9
    k = 0 # final sum
    if t1[0] == t2[0]: # if labels are the same
        max_tree_len = min(len(t1[1]),len(t2[1]))
        for tree_len in range(1,max_tree_len+1) :
            # length of all possible subtrees 
            #ChildSeqs = GetAllComb(tree_len,[c for c in range(min(len(t1[1]),len(t2[1])))])
            if max_tree_len <= 10:
                ChildSeqs = comb_dict[max_tree_len][tree_len]
            else:
                ChildSeqs = GetAllComb(tree_len,[c for c in range(max_tree_len)])
            #print(ChildSeqs)
            # list of indices of all possible child sequences of given length
            for j1 in ChildSeqs:
                for j2 in ChildSeqs:
                    # Get all pairs of sub sequences
                    #print(j1,j2)
                    prod = 1 # product
                    #print(len(j1),tree_len)
                    for i in range(tree_len):
                        childt1 = tree1[1][j1[i]] # token object
                        childt2 = tree2[1][j2[i]]
                        prodt1 = (childt1,[j for j in childt1.children if j.text.isalpha()]) # build subtree 
                        prodt2 = (childt2,[j for j in childt2.children if j.text.isalpha()])
                        #if prodt1[1] and prodt2[1]:
                        #print(prodt1,prodt2)
                        t1index = tree1Index[childt1]
                        t2index = tree2Index[childt2]
                        res = KernelFunction(prodt1,prodt2,t1index,t2index,prev,tree1Index,tree2Index) 
                        prod = prod * res
                        #if prev[t1index][t2index]  == -1:
                        prev[t1index][t2index] = res
                        #print(prod)
                        # continue matching subtree
                    # finish matching indices j1 and j2, sum 
                    dt1 = j1[-1] - j1[0] + 1
                    dt2 = j2[-1] - j2[0] + 1    
                    k += ((lambda_) ** (dt1 + dt2)) * prod
        return mu * (lambda_**2 + k)
    else:
        return 0
    
def PartialTreeKernel(tree1,tree2):
    prev = [[-1 for i in range(len(tree2))] for j in range(len(tree1))]
    tree1Index = {list(tree1)[i]:i for i in range(len(tree1))}
    tree2Index = {list(tree2)[i]:i for i in range(len(tree2))}
    sim = 0
    for tokent1 in tree1:
        prodt1 = (tokent1,[i for i in tokent1.children if i.text.isalpha()])
        for tokent2 in tree2:
            # sum over all nodes
            prodt2 = (tokent2,[i for i in tokent2.children if i.text.isalpha()])
            if tokent1.text.isalpha() and (tokent1.dep_ == tokent2.dep_):
                #print('subtree',prodt1,prodt2)
                t1index = tree1Index[tokent1]
                t2index = tree2Index[tokent2]
                res = KernelFunction(prodt1,prodt2,t1index,t2index,prev,tree1Index,tree2Index)
                sim += res
                prev[t1index][t2index] = res
    return sim 

def normPTK(tree1,tree2):
    return PartialTreeKernel(tree1,tree2)/math.sqrt(PartialTreeKernel(tree1,tree1)*PartialTreeKernel(tree2,tree2))



# Extract vac
def GetVAC(text,render=False):
    VACs = []
    doc = NLP(text)
    VOffsets = [0 for i in range(len(doc))]
    for t in doc:
        if t.pos_ == 'VERB':
            VOffsets[t.i] = 1
    matches = matcher(doc) 
    
    for m in matches: # for overlapping matches
        RuleName = RuleID[m[0]]
        SpanStart = min(m[1])
        SpanEnd = max(m[1])
        VerbIndex = m[1][0]
        if RuleName == 'v_noun(dobj)':
            for token in doc[VerbIndex].children: 
                if token.dep_ in ['ccomp','acomp','xcomp','dative','oprd']:
                    break
            else:
                VOffsets[VerbIndex] = 0
                VACs.append((RuleName.replace('v_',doc[VerbIndex].lemma_+'_'), SpanStart,SpanEnd+1))
        elif RuleName == 'v_to_v2':
            for token in doc[VerbIndex].children: 
                if token.dep_ in  ['dobj','acomp']:
                    break
                if token.dep_ == 'ccomp':
                    for cc in token.children:
                        if cc.dep_ in ['nsubj','csubj'] or cc.pos_ == 'SCONJ':
                            break
            else:
                VOffsets[m[1][0]] = 0
                VACs.append((RuleName.replace('v_',doc[VerbIndex].lemma_+'_'), SpanStart,SpanEnd+1))
        elif RuleName == 'v_noun(dobj)_to_v2':
            for token in doc[m[1][2]].lefts: 
                if str(token) in ["that",'how','what','where','why','when','who']:
                    break
            else:
                VOffsets[m[1][0]] = 0
                VACs.append((RuleName.replace('v_',doc[VerbIndex].lemma_+'_'), SpanStart,SpanEnd+1))
        elif RuleName == 'v_wh(objcl)' or RuleName == 'v_that(objcl)':
            for token in doc[VerbIndex].rights: 
                if token.dep_ == 'dobj':
                    break
            else:
                VOffsets[m[1][0]] = 0
                VACs.append((RuleName.replace('v_',doc[VerbIndex].lemma_+'_'), SpanStart,SpanEnd+1))
        elif RuleName == 'v_to_do':
            for token in doc[VerbIndex].rights: 
                if token.pos_ in ["ADJ"]:
                    break
            else:
                VOffsets[VerbIndex] = 0
                VACs.append((RuleName.replace('v_',doc[VerbIndex].lemma_+'_'), SpanStart,SpanEnd+1))
        elif RuleName in ['It_is_?_that/who_v_','It_is_?_to_v_']:
            VACs.append((RuleName.replace('v_',doc[VerbIndex].lemma_+'_'), SpanStart,SpanEnd+1))
        else:
            VOffsets[VerbIndex] = 0
            VACs.append((RuleName.replace('v_',doc[VerbIndex].lemma_+'_'), SpanStart,SpanEnd+1))

    for j in range(len(VOffsets)):
        if VOffsets[j] == 1:
            for token in doc[j].rights:
                if token.dep_ == 'ccomp' and token.pos_ in ['AUX','VERB']:
                    for cc in token.lefts:
                        if str(cc) in ["that",'how','what','where','why','when','who']:
                            break
                        if cc.dep_ == 'nsubj':
                            VACs.append(('{}_cl(obj)'.format(doc[j].lemma_),j,token.i+1))
                            VOffsets[j] = 0
                            break
                if VOffsets[j] == 0:
                    break
        
        if VOffsets[j] == 1:
            vi = True
            children = doc[j].children
            for token in children:
                if token.dep_ in ['dobj','ccomp','acomp','xcomp']:
                    vi = False
            for token in doc[j].rights:
                if token.dep_ == 'prep' and vi:
                    for cc in token.rights:
                        if cc.dep_ == 'pcomp' or cc.dep_ == 'pobj':
                            VACs.append(('{}_vi_prep'.format(doc[j].lemma_),j,cc.i+1))
                            VOffsets[j] = 0
                            break
                    break
            if vi and VOffsets[j] == 1:
                VACs.append(('{}_vi'.format(doc[j].lemma_),j,j+1))
                VOffsets[j] = 0
            if VOffsets[j] == 1 and 'dobj' in [cc.dep_ for cc in doc[j].lefts]:
                VOffsets[j] = 0
                VACs.append(('{}_noun(obj)'.format(doc[j].lemma_),j-1,j+1))
            if not vi and VOffsets[j] == 1:
                VACs.append(('other_{}_'.format(doc[j].lemma_),j,j+1))
    doc.spans['sc'] = []
    for vac in VACs:
        doc.spans['sc'].append(Span(doc,vac[1],vac[2],vac[0]))
    if render:
        displacy.render(doc,style='span')
    else:
        return VACs



# Get dependency trees from spacy to nltk
def CreateDepTree(token):
    if token.n_lefts == 0 and token.n_rights == 0:
        return nltk.Tree(token.dep_,[])
    else:    
        return nltk.Tree(token.dep_,[CreateDepTree(child) for child in token.children])
    
    
    
    
    
# class defnition
## Dataloader
class DataLoader():
    def __init__(self) :
        # dict for path and source
        self.FilePath = {
            'L2Writing':'data/L2writing/train.csv',
            'SHU':'data/SHU/spring.xlsx',
            'ASAP':'data/asap-aes/training_set_rel3.xls',
            'final':'data/Final-all/winter.csv'
        }
    def LoadData(self,source):
        self.source = source
        self.FileName = self.FilePath[self.source]
        if 'csv' in self.FileName:
            self.data = pd.read_csv(self.FileName)
        else:
            self.data = pd.read_excel(self.FileName)
    def GetData(self,source):
        '''
        source: one of 'L2Writing','SHU','ASAP','final'
        '''
        self.LoadData(source)
        return self.data
    def ShuffleData(self,seed):
        self.shuffled = shuffle(self.data,random_state=seed)
    def GetShuffled(self,seed):
        self.ShuffleData(seed)
        return self.shuffled
    


# Get bert embeddigns
class GetBERTEmbeddings():
    def __init__(self,input,model):
        # Check input
        if model in ['model/deberta-v3-large','model/bigbird-large','model/deberta-v3-base','model/deberta-base']:
            self.input = input
            # load model and tokenizer
            self.model = AutoModel.from_pretrained(model)
            self.model.config.hidden_dropout = 0.
            self.model.config.hidden_dropout_prob = 0.
            self.model.config.attention_dropout = 0.
            self.model.config.attention_probs_dropout_prob = 0.
            self.tokenizer = AutoTokenizer.from_pretrained('model/tokenizer/{}'.format(model.split('/')[1]))
            self.hidden = []
        else:
            raise KeyError
    def tokenize(self,SeqLen=1024):
        self.tokenized = []
        for seq in self.input:
            self.tokenized.append(
                self.tokenizer(seq,
                    add_special_tokens=True,
                    max_length=SeqLen,    # max sequence length, default is 1024
                    return_tensors='pt',  # return in tensor
                    padding='max_length',
                    truncation=True))
        self.input = self.tokenized # move to gpu
        self.input = [i.to('cuda') for i in self.input]
    def inf(self,stop=1000,SeqLen = 1024):
        self.tokenize(SeqLen=SeqLen)
        print('tokenized')
        self.model = self.model.to('cuda') # move model to gpu
        for run in range(len(self.input) // stop):
            for i in range(stop):
                with torch.no_grad():
                    out = self.model(self.input[run*stop+i]['input_ids'],self.input[run*stop+i]['attention_mask']) # inference
                self.hidden.append(out.last_hidden_state.detach().cpu()) # detach to cpu
                if i % 10 == 0:
                    print('{}/{}, run:{}'.format(i,stop,run))
                del out 
            torch.cuda.empty_cache() # clear cuda memory for next run
        # remaining ones
        t = len(self.input) // stop * stop
        for i in range(t,len(self.input)):
            with torch.no_grad():
                out = self.model(self.input[i]['input_ids'],self.input[i]['attention_mask']) # inference
            self.hidden.append(out.last_hidden_state.detach().cpu()) # detach to cpu
            if i % 10 == 0:
                print('{}/{}, run:{}'.format(i,stop,'f'))
            del out 
        torch.cuda.empty_cache()
    def CLSEmbedding(self,i):
        self.CLS = self.hidden[i][:,0].detach().clone().cpu()
        return self.CLS.clone()
    def MaxPooling(self,i):
        hidden = self.hidden[i]
        self.AttentionMask = self.tokenized[i]['attention_mask'].unsqueeze(-1).expand(hidden.size())
        hidden[self.AttentionMask == 0] = -1e9 # ignore paddings
        self.MaxP = torch.max(hidden,1)[0]
        return self.MaxP
    def MeanPooling(self,i):
        hidden = self.hidden[i].detach().cpu()
        self.AttentionMask = self.tokenized[i]['attention_mask'].unsqueeze(-1).expand(hidden.size()).detach().cpu()
        SumEmbeddings = torch.sum(self.AttentionMask*hidden,1) # ignore paddings
        SumMask = self.AttentionMask.sum(1)
        SumMask = SumMask.clamp(min=1e-9) # prevents division by zero
        self.MeanP = SumEmbeddings/SumMask
        return self.MeanP
    def GetEmbeddings(self,type) :
        EmbeddingType = {
            'CLS':self.CLSEmbedding,
            'MaxP':self.MaxPooling,
            'MeanP':self.MeanPooling
        }
        result = [EmbeddingType[type](i) for i in range(len(self.input)) ]
        return result



# Get linguistic features
class FeatureExtraction():
    def __init__(self,text) :
        self.text = text 
        self.features = {} # final result
        self.counts = {} # stores all counts in one traversal
        # initialize pos, phrase, dep counts
        for tag in AllPosTagList + PhraseList + DepTagList + TenseList:
            self.counts[tag] = 0 
        # sets for unique words
        self.unique_words ={'VERB':set(),'NOUN':set(),'ADJ':set(),'ADV':set()}
        # initialize pos counts
    def nlp(self): # executes spacy pipeline
        self.doc = NLP(self.text)

    def traverse(self):
        # features extracted with spacy
        sent_count = 0
        token_count = 0
        total_tree_height = 0
        # sentence count
        self.sents = [j for j in self.doc.sents if len(str(j).split()) > 3]
        for sent in self.sents:
            sent_count += 1
            # token count
            # assemble sentence
            sent_token = []
            for token in sent:
                # get attributes
                word = token.text
                pos = token.pos_
                dep = token.dep_
                if word.isalpha():
                    sent_token.append(word) # assemble sentence
                    token_count += 1
                    # pos count
                    if pos in AllPosTagList :
                        self.counts[pos] += 1
                    # store unique words
                    if pos in self.unique_words:
                        self.unique_words[pos].add(token.text)
                    if dep in DepTagList:
                        self.counts[dep] += 1
                    if token.pos_ =='VERB':
                        m = token.morph.to_dict()
                        if 'Tense' in m:
                            self.counts[m['Tense']] += 1
                        if 'Aspect' in m:
                            self.counts[m['Aspect']] += 1
            # finish token traversal
            if len(sent_token) == 0:
                continue 
        # features extracted with parser
            # parsing
            sent_tree = str(SuPar.predict(sent_token,prob=True,verbose= False).sentences[0])
            # create tree
            nltk_tree = nltk.Tree.fromstring(sent_tree)
            total_tree_height += nltk_tree.height()
            # Find phrases
            for phrase in PhraseList:
                try:
                    self.counts[phrase] += FindPhrase(nltk_tree,phrase)
                except:
                    pass
        # finish sentence traversal



        # counts
        self.counts['sent'] = sent_count # sentence count
        self.counts['token'] = token_count # token count
        self.counts['content_word'] = sum([self.counts[t] for t in CWPosTagList]) # content words count
        self.counts['function_word'] = sum([self.counts[t] for t in FWPoSTagList]) # function words count
        # unique word counts
        self.counts['unique_verb'] = len(self.unique_words['VERB'])
        self.counts['unique_noun'] = len(self.unique_words['NOUN'])
        self.counts['unique_adj'] = len(self.unique_words['ADJ'])
        self.counts['unique_adv'] = len(self.unique_words['ADV'])
        for i in self.counts.keys():
            if self.counts[i] == 0:
                self.counts[i] = 1
        # token features
        self.features['token/sent'] = self.counts['token']/self.counts['sent']
        # pos features:
        self.features['verb/sent'] = self.counts['VERB']/self.counts['sent']
        self.features['noun/sent'] = self.counts['NOUN']/self.counts['sent']
        self.features['adj/sent'] = self.counts['ADJ']/self.counts['sent']
        self.features['adv/sent'] = self.counts['ADV']/self.counts['sent']
        
        self.features['verb/token'] = self.counts['VERB']/self.counts['token']
        self.features['noun/token'] = self.counts['NOUN']/self.counts['token']
        self.features['adj/token'] = self.counts['ADJ']/self.counts['token']
        self.features['adv/token'] = self.counts['ADV']/self.counts['token']

        self.features['content/function'] = self.counts['content_word']/self.counts['function_word']
        # variety features
        self.features['verb_var'] = self.counts['unique_verb']/math.sqrt(2*self.counts['VERB'])
        self.features['noun_var'] = self.counts['unique_noun']/math.sqrt(2*self.counts['NOUN'])
        self.features['adj_var'] = self.counts['unique_adj']/math.sqrt(2*self.counts['ADJ'])
        self.features['adv_var'] = self.counts['unique_adv']/math.sqrt(2*self.counts['ADV'])
        # Tree height
        self.features['tree_h/sent'] = total_tree_height/self.counts['sent']
        # phrasal features
        self.features['vp/sent'] = self.counts['VP']/self.counts['sent']
        self.features['np/sent'] = self.counts['NP']/self.counts['sent']
        self.features['adjp/sent'] = self.counts['ADJP']/self.counts['sent']
        self.features['advp/sent'] = self.counts['ADVP']/self.counts['sent']
        self.features['pp/sent'] = self.counts['PP']/self.counts['sent']
        # dependency and tense features
        for tag in DepTagList + TenseList:
            self.features['{}/sent'.format(tag)] = self.counts[tag]/self.counts['sent']
    def process(self):
        self.nlp()
        self.traverse()
    def get_all_feature_names(self):
        return list(self.features.keys()) + list(self.counts.keys())
    def get_data(self):
        result = []
        for i in range(68):
            result.append(self.features[FeatureNameList[i]])
        for i in range(68,143):
            result.append(self.counts[FeatureNameList[i]])
        return result