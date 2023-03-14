from transformers import AutoModel, AutoTokenizer
import torch

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
