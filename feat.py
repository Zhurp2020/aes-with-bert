# Get fine-grained syntactic features

import spacy
import nltk
from supar import Parser
from math import sqrt
from thinc.api import set_gpu_allocator, require_gpu



set_gpu_allocator("pytorch")
require_gpu(0)



NLP = spacy.load('en_core_web_trf')
SuPar = Parser.load('crf-con-en')


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
        for sent in self.doc.sents:
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
        self.features['verb_var'] = self.counts['unique_verb']/sqrt(2*self.counts['VERB'])
        self.features['noun_var'] = self.counts['unique_noun']/sqrt(2*self.counts['NOUN'])
        self.features['adj_var'] = self.counts['unique_adj']/sqrt(2*self.counts['ADJ'])
        self.features['adv_var'] = self.counts['unique_adv']/sqrt(2*self.counts['ADV'])
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

