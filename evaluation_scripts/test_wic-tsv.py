
import sys
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from sklearn import metrics
from src.helper import *

def eval_wic_cosine(scores_pred,golds,thres=None):
    scores_pred,golds=np.array(scores_pred),np.array(golds)
    if thres:
        scores_pred_label = np.array(['F'] * len(scores_pred))
        scores_true_indexes = np.where(scores_pred > thres)

        scores_pred_label[scores_true_indexes] = 'T'
        corrects_true = np.where((np.array(scores_pred_label) == 'T') & (np.array(golds) == 'T'))[0]
        corrects_false = np.where((np.array(scores_pred_label) == 'F') & (np.array(golds) == 'F'))[0]
        num_corrects = len(corrects_true) + len(corrects_false)


        print ('==WIC RESULTS==: thres: {0}, num of correct: {1}, percentage: {2}'.format(thres,num_corrects,num_corrects/len(scores_pred)))

    else:
        thres=thres_search(scores_pred,golds)
        thres,scores_pred_label=eval_wic_cosine(scores_pred,golds,thres)
    return thres,scores_pred_label

def thres_search(scores_pred,golds):

    thres=scores_pred[np.argmax(scores_pred)]
    thres_min=scores_pred[np.argmin(scores_pred)]
    num_corrects_prevmax=-1
    num_corrects=0
    thres_max=0
    while thres>=thres_min:
        if num_corrects>num_corrects_prevmax:
            num_corrects_prevmax=num_corrects
            thres_max=thres
        scores_pred_label = np.array(['F'] * len(scores_pred))
        thres-=0.01

        scores_true_indexes = np.where(scores_pred>thres)

        scores_pred_label[scores_true_indexes]='T'
        corrects_true = np.where((np.array(scores_pred_label) == 'T') & (np.array(golds) == 'T'))[0]
        corrects_false=np.where((np.array(scores_pred_label) == 'F') & (np.array(golds) == 'F'))[0]
        num_corrects=len(corrects_true)+len(corrects_false)
        # print ('thres: {0}, num of correct: {1}, percentage is" {2}'.format(thres,num_corrects,num_corrects/len(scores_pred)))
    return thres_max


def wic_scores(lines,tokenizer,model,flag,layer_start,layer_end,maxlen):
    src,tgt,label=list(zip(*lines))
    src,tgt,label=list(src),list(tgt),list(label)
    string_features1, string_features2 = [], []
    if flag.startswith('token'):
        for i,line in enumerate(tgt):
            w=src[i].split()[src[i].split().index('[')+1]
            if basename=='wic_tsv_def_wic':
                line='[ '+w+' ] means '+line
            elif basename=='wic_tsv_hyp_wic':
                line='[ '+w+' ] is a kind of '+','.join(line.split())
            elif basename=='wic_tsv_def_hyp_wic':
                hyp=line.split('||')[1].split()
                linehyp='[ '+w+' ] is a kind of '+','.join(hyp)
                defi=line.split('||')[0]
                linedefi=' and means '+defi
                line=linehyp+linedefi
            tgt[i]=line
    
    for i in tqdm(np.arange(0, len(src), bsz)):
        np_feature_mean_tok=get_embed(src[i:i+bsz],tokenizer,model,flag,layer_start,layer_end,maxlen)
        string_features1.append(np_feature_mean_tok)
    string_features1_stacked =  np.concatenate(string_features1, 0)
    for i in tqdm(np.arange(0, len(tgt), bsz)):
        np_feature_mean_tok=get_embed(tgt[i:i+bsz],tokenizer,model,flag,layer_start,layer_end,maxlen)
        string_features2.append(np_feature_mean_tok)
    string_features2_stacked =  np.concatenate(string_features2, 0)
    string_features1_stacked,string_features2_stacked=torch.from_numpy(string_features1_stacked),torch.from_numpy(string_features2_stacked)
    scores_pred=produce_cosine_list(string_features1_stacked,string_features2_stacked)
    return scores_pred,label

if __name__=='__main__':
    
    
    bsz = 128
    maxlen = 90 # 64

    model_name=sys.argv[1]
    datadir=sys.argv[2]
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

    basename=os.path.basename(datadir)
    wic_train=os.path.join(datadir,'train.tsv')
    wic_test=os.path.join(datadir,'test.tsv')
    wic_dev=os.path.join(datadir,'dev.tsv')

    test_lines=[line.strip().split('\t') for line in open(wic_test).readlines()[1:]]
    dev_lines=[line.strip().split('\t') for line in open(wic_dev).readlines()[1:]]
    train_lines=[line.strip().split('\t') for line in open(wic_train).readlines()[1:]]

    dev_scores_pred,dev_label=wic_scores(dev_lines,tokenizer,model,flag,layer_start,layer_end,maxlen)
 
    print ('=======dev set accuracy=======')
    dev_thres, dev_pred=eval_wic_cosine(dev_scores_pred,dev_label,thres=None)

    print ('=======test set accuray=======')
    print ('For WiC and WiC-tsv, the result here is a placeholder. \n You need to upload the predicted test file to their codalab competition pages.')
    test_scores_pred,test_label=wic_scores(test_lines,tokenizer,model,flag,layer_start,layer_end,maxlen)
    _, test_pred=eval_wic_cosine(test_scores_pred,test_label,thres=dev_thres)
   
    print ('=======auc=======')
    fpr, tpr, thresholds = metrics.roc_curve(dev_label, dev_scores_pred, pos_label='T')    
    print ('dev auc: ',metrics.auc(fpr, tpr))
   

    with open(wic_test+'.preds','w') as f:
        for score in test_pred:
            f.write(str(score)+'\n')
