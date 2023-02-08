import spacy
from lingfeat import extractor
class FeatureExtraction():
    def __init__(self,text) :
        self.text = text 
        self.extractor = extractor.pass_text(text)
        self.count = self.extractor.preprocess()
        self.doc = self.extractor.NLP_doc
    def GetPos(self):
        d = self.extractor.POSF_()
        feats = [
            d['at_NoTag_C'], #average count of Noun POS tags per token
            d['at_VeTag_C'], #average count of Verb POS tags per token
            d['at_AjTag_C'], #average count of Adjective POS tags per token
            d['at_AvTag_C'], #average count of Adverb POS tags per token
            d['ra_CoFuW_C'],  #ratio of Content words to Function words
        ]
        return feats
    def GetVar(self):
        d = self.extractor.VarF_()
        feats = [
            d['CorrNoV_S'], #unique Nouns/sqrt(2*total Nouns)
            d['CorrVeV_S'], #unique Verbs/sqrt(2*total Verbs)
            d['CorrAjV_S'], #unique Adjectives/sqrt(2*total Adjectives) (Corrected Adjective
            d['CorrAvV_S'], #unique Adverbs/sqrt(2*total Adverbs) (Corrected AdVerb Variation-1)
        ]
        return feats
    def GetTree(self):
        d = self.extractor.TrSF_()
        feats = [
            d['as_TreeH_C'], #average Tree height per sentence
            d['at_FTree_C']# average length of flattened Trees per token (word)
        ]
        return feats
    def GetPhr(self):
        d = self.extractor.PhrF_()
        feats = [
            d['as_NoPhr_C'], #average count of Noun phrases per sentence
            d['as_VePhr_C'],# average count of Verb phrases per sentence
            d['as_PrPhr_C'], #average count of prepositional phrases per sentence
            d['as_AjPhr_C'], #average count of Adjective phrases per sentence
            d['as_AvPhr_C']# 	average count of Adverb phrases per sentence
        ]
        return feats
    def GetTenDep(self):
        d = {
        'Pres':0,
        'Prog':0,
        'Past':0,
        'Perf':0,
        'advcl':0,
        'acl':0,
        'agent':0,
        'aux':0,
        'auxpass':0,
        'ccomp':0,
        'pcomp':0,
        'cc':0,
        'xcomp':0,
        'csubj':0,
        'csubjpass':0
            }
        for sent in self.doc.sents:
            for t in sent:
                if t.pos_ =='VERB':
                    m = t.morph.to_dict()
                    if 'Tense' in m:
                        d[m['Tense']] += 1
                    if 'Aspect' in m:
                        d[m['Aspect']] += 1
                if t.dep_ in d:
                    d[t.dep_] += 1
        n = self.count['n_sent']
        f = [d[i]/n for i in d.keys()]
        return f
        '''
    def GetDep(self):
        d = {
        'advcl':0,
        'acl':0,
        'agent':0,
        'aux':0,
        'auxpass':0,
        'ccomp':0,
        'pcomp':0,
        'cc':0,
        'xcomp':0,
        'csubj':0,
        'csubjpass':0
            }
        for sent in self.doc.sents:
            for t in sent:
                if t.dep_ in d:
                    d[t.dep_] += 1
        n = self.count['n_sent']
        f = [d[i]/n for i in d.keys()]
        return f'''
    def GetFine(self):
        return self.GetPos() + self.GetPhr() + self.GetVar() + self.GetTenDep() + self.GetTree()