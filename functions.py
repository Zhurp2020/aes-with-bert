import pandas as pd
import numpy as np
import math

from sklearn.utils import shuffle
from sklearn import linear_model 
from sklearn.metrics import cohen_kappa_score,mean_absolute_error,mean_squared_error,accuracy_score,explained_variance_score,r2_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE,MDS
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans,SpectralClustering,AffinityPropagation,AgglomerativeClustering
from scipy.stats import ttest_ind
import statsmodels.api as sm
import markov_clustering as mcl
import xgboost as xgb
import pingouin as pg

import networkx as nx

from transformers import AutoModel, AutoTokenizer
import torch
from thinc.api import set_gpu_allocator, require_gpu

import spacy
from spacy import displacy
from textacy import preprocessing

import nltk
from supar import Parser

import matplotlib.pyplot as plt
import seaborn as sns


set_gpu_allocator("pytorch")
require_gpu(0)

NLP = spacy.load('en_core_web_trf')
SuPar = Parser.load('crf-con-en')


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
    def ShuffleData(self):
        self.shuffled = shuffle(self.data)
    def GetShuffled(self):
        self.ShuffleData()
        return self.shuffled
    


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









DepTagList = [i.strip() for i in '''acl, acomp, advcl, advmod, agent, amod, appos, attr, aux, auxpass, case, cc, ccomp, compound, conj, csubj, csubjpass, dative, dep, det, dobj, expl, intj, mark, meta, neg, nmod, npadvmod, nsubj, nsubjpass, nummod, oprd, parataxis, pcomp, pobj, poss, preconj, predet, prep, prt, punct, quantmod, relcl, xcomp'''.split(',')]

# Content word pos tags
CWPosTagList = ["NOUN","VERB","PRPON","INTJ","ADJ","ADV"]
# Function word pos tags
FWPoSTagList = ['ADP','AUX','CCONJ','DET','NUM','PART','PRON','SCONJ']
AllPosTagList = CWPosTagList + FWPoSTagList
'''
ADJ: adjective
ADP: adposition
ADV: adverb
AUX: auxiliary
CCONJ: coordinating conjunction
DET: determiner
INTJ: interjection
NOUN: noun
NUM: numeral
PART: particle
PRON: pronoun
PROPN: proper noun
PUNCT: punctuation
SCONJ: subordinating conjunction
SYM: symbol
VERB: verb
X: other
'''

# Phrases to look for
PhraseList = ['VP','NP','ADJP','PP','ADVP']
TenseList = ['Pres', 'Prog','Past','Perf']

FeatureNameList = ['token/sent', 'verb/sent', 'noun/sent', 'adj/sent', 'adv/sent', 'verb/token', 'noun/token', 'adj/token', 'adv/token', 'content/function', 'verb_var', 'noun_var', 'adj_var', 'adv_var', 'tree_h/sent', 'vp/sent', 'np/sent', 'adjp/sent', 'advp/sent', 'pp/sent', 'acl/sent', 'acomp/sent', 'advcl/sent', 'advmod/sent', 'agent/sent', 'amod/sent', 'appos/sent', 'attr/sent', 'aux/sent', 'auxpass/sent', 'case/sent', 'cc/sent', 'ccomp/sent', 'compound/sent', 'conj/sent', 'csubj/sent', 'csubjpass/sent', 'dative/sent', 'dep/sent', 'det/sent', 'dobj/sent', 'expl/sent', 'intj/sent', 'mark/sent', 'meta/sent', 'neg/sent', 'nmod/sent', 'npadvmod/sent', 'nsubj/sent', 'nsubjpass/sent', 'nummod/sent', 'oprd/sent', 'parataxis/sent', 'pcomp/sent', 'pobj/sent', 'poss/sent', 'preconj/sent', 'predet/sent', 'prep/sent', 'prt/sent', 'punct/sent', 'quantmod/sent', 'relcl/sent', 'xcomp/sent', 'Pres/sent', 'Prog/sent', 'Past/sent', 'Perf/sent', 'NOUN', 'VERB', 'PRPON', 'INTJ', 'ADJ', 'ADV', 'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ', 'VP', 'NP', 'ADJP', 'PP', 'ADVP', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'predet', 'prep', 'prt', 'punct', 'quantmod', 'relcl', 'xcomp', 'Pres', 'Prog', 'Past', 'Perf', 'sent', 'token', 'content_word', 'function_word', 'unique_verb', 'unique_noun', 'unique_adj', 'unique_adv']


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


def CreateDepTree(token):
    if token.n_lefts == 0 and token.n_rights == 0:
        return nltk.Tree(token.dep_,[])
    else:    
        return nltk.Tree(token.dep_,[CreateDepTree(child) for child in token.children])