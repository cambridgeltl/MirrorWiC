
import sys
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from src.helper import *
from scipy.stats import spearmanr




def usim_predict(left,right,tokenizer,model,flag,layer_start,layer_end,maxlen):
    string_features1, string_features2 = [], []
    for i in tqdm(np.arange(0, len(left), bsz)):
        np_feature_mean_tok=get_embed(left[i:i+bsz],tokenizer,model,flag,layer_start,layer_end,maxlen)

        string_features1.append(np_feature_mean_tok)
    string_features1_stacked =  np.concatenate(string_features1, 0)
    for i in tqdm(np.arange(0, len(right), bsz)):
        np_feature_mean_tok=get_embed(right[i:i+bsz],tokenizer,model,flag,layer_start,layer_end,maxlen)

        string_features2.append(np_feature_mean_tok)
    string_features2_stacked =  np.concatenate(string_features2, 0)
    string_features1_stacked,string_features2_stacked=torch.from_numpy(string_features1_stacked),torch.from_numpy(string_features2_stacked)
    scores_pred=produce_cosine_list(string_features1_stacked,string_features2_stacked)
    return scores_pred


if __name__=='__main__':
    
    

   
    bsz = 128
    maxlen = 80 # 64

    model_name=sys.argv[1]
    data=sys.argv[2]
    cuda=sys.argv[4]
    flag=sys.argv[3]
    layers,layer_start,layer_end=None,None,None
    if flag.startswith('token') or flag=='mean':
        layers=sys.argv[5]
        layer_start,layer_end=int(layers.split('~')[0]),int(layers.split('~')[1])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.cuda()
    model.eval()


    # data=os.path.join(datadir,'usim_en.txt')
    lines_left=[]
    lines_right=[]
    golds=[]
    for i,line in enumerate(open(data).readlines()):
        context,wid,rating=line.split('\t')[:3]
        wlst=context.split()
        w=wlst[int(wid)]
        line=' '.join(wlst[:int(wid)]).replace(']','').replace('[','')+' [ '+w+' ] '+' '.join(wlst[int(wid)+1:]).replace(']','').replace('[','')
        
        if i%2==0:
            lines_left.append(line)
            golds.append(float(rating))
        else:
            lines_right.append(line)
            assert float(rating)==golds[-1]
    assert len(lines_right)==len(lines_left)==len(golds)
    preds=usim_predict(lines_left,lines_right,tokenizer,model,flag,layer_start,layer_end,maxlen)
    rho=spearmanr(np.array(preds),np.array(golds))
    print (rho)

  
