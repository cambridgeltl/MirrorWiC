from collections import defaultdict
from random import sample
import argparse
import os
from copy import deepcopy
from evaluation_scripts.src.helper import lg2wordtokenize
from transformers import AutoTokenizer
import numpy as np

def erase_and_mask(s, erase_len):
    if erase_len==0: return s
    if len(s) <= erase_len: return s
    if len(s) < 10: return s # if too short, no augmentation
    ind = np.random.randint(len(s)-erase_len)
    left, right = s.split(s[ind:ind+erase_len], 1)
    return " ".join([left, "[MASK]", right])

def erase_and_mask_on_wic(example,erase_len):
    w=' '.join(example[example.index('['):example.index(']')+1])
    sent_prev=erase_and_mask(' '.join(example[:example.index('[')]),erase_len)
    sent_after=erase_and_mask(' '.join(example[example.index(']')+1:]),erase_len)
    example=sent_prev+' '+w+' '+sent_after
    return example

def wic_transform(sent,sentnum,word_tokenize,word_position_max=20):
    sent=sent.replace('[','').replace(']','')
    w2sentsnew=defaultdict(list)
    wlist=word_tokenize(sent)
    for i,w in enumerate(wlist[:word_position_max]):
        # if w.isalpha():
            if w.strip() and tokenizer1.tokenize(w,add_special_tokens=False) and tokenizer2.tokenize(w,add_special_tokens=False):
                sentorig=deepcopy(wlist)
                sentorig.insert(i,'[')
                sentorig.insert(i+2,']')
                assert sentorig[i+1:i+2]==[w]
                w2sentsnew[w].append(sentorig)
    if sentnum>len(w2sentsnew):
        sentnum=len(w2sentsnew)
    wlist=sample(list(w2sentsnew.keys()),sentnum)
    sentsnew=[]
    for w  in wlist:
        sentsnew.append(sample(w2sentsnew[w],1)[0])
    return sentsnew



parser = argparse.ArgumentParser(description='get_mirrorwic_traindata')
parser.add_argument('--data',type=str,help='data with one sentence per line')
parser.add_argument('--lg',type=str,help='language')
parser.add_argument('--random_er',type=int,help='random erasuer length')
args=parser.parse_args()

print (args.data)
maxlen=150
sentnum=1 # the number of wic example from the original sentence
word_tokenize,_=lg2wordtokenize(args.lg)
tokenizer1=AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
tokenizer2=AutoTokenizer.from_pretrained('bert-base-uncased')
lines=[line.strip() for line in open(args.data) if len(line)>10 and len(line)<maxlen]
lines=list(set(lines))

fname=os.path.basename(args.data)+'.mirror.wic.re{0}'.format(str(args.random_er))

with open(os.path.join(os.path.dirname(args.data),fname),'w') as f:
    for i,line in enumerate(lines):
        line=line.replace('||','//')
        examples=wic_transform(line,sentnum,word_tokenize)
        for example in examples:
            example_masked=erase_and_mask_on_wic(example, args.random_er)
            f.write('||'.join([args.lg+str(i),' '.join(example),example_masked])+'\n')
        