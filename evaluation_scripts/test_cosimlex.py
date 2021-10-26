def clean_sent(sent):
    return sent.replace('<strong>','').replace('</strong>','').replace('[','').replace(']','').strip()

def right(wlst,ide):
    if ide>=len(wlst):
        return []
    else:
        return wlst[ide:]
if __name__ =='__main__':

    import sys
    from src.helper import * 
    from collections import defaultdict
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm.auto import tqdm
    from nltk.tokenize import wordpunct_tokenize as tokenize
    import os

    bsz = 128
    maxlen = 120 # 64
    model_name=sys.argv[1]
    data=sys.argv[2]
    flag=sys.argv[3]
    cuda=sys.argv[4]
    layers,layer_start,layer_end=None,None,None
    if flag.startswith('token') or flag=='mean':
        layers=sys.argv[5]
        layer_start,layer_end=int(layers.split('~')[0]),int(layers.split('~')[1])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.cuda()
    model.eval()

    scores=[]
    for line in tqdm(open(data).readlines()[1:]):
        _,_,c1,c2,_,_,_,_=line.split('\t')
        c1=c1.replace('[','').replace(']','')
        c2=c2.replace('[','').replace(']','')
        # print (c1)
        # print (c2)
        c1=tokenize(c1.replace('<strong>',' [').replace('</strong>','] '))
        c2=tokenize(c2.replace('<strong>',' [').replace('</strong>','] '))

        target1s=[i for i,w in enumerate(c1) if c1[i-1]=='[']
        target2s=[i for i,w in enumerate(c2) if c2[i-1]=='[']
        try:
            assert len(target1s)==len(target2s)==2
        except AssertionError as e:
            print (target2s,target2s)
            print (c1)
            print (c2)
        c1_w1=clean_sent(' '.join(c1[:target1s[0]]))+' [ '+c1[target1s[0]]+' ] '+clean_sent(' '.join(right(c1,target1s[0]+1)))
        c1_w2=clean_sent(' '.join(c1[:target1s[1]]))+' [ '+c1[target1s[1]]+' ] '+clean_sent(' '.join(right(c1,target1s[1]+1)))
        c2_w2=clean_sent(' '.join(c2[:target2s[1]]))+' [ '+c2[target2s[1]]+' ] '+clean_sent(' '.join(right(c2,target2s[1]+1)))
        c2_w1=clean_sent(' '.join(c2[:target2s[0]]))+' [ '+c2[target2s[0]]+' ] '+clean_sent(' '.join(right(c2,target2s[0]+1)))

        embed_src=get_embed([c1_w1,c2_w1],tokenizer,model,flag,layer_start,layer_end,maxlen)
        embed_tgt=get_embed([c1_w2,c2_w2],tokenizer,model,flag,layer_start,layer_end,maxlen)
        embed_src,embed_tgt=torch.from_numpy(embed_src),torch.from_numpy(embed_tgt)
        scores_pred=produce_cosine_list(embed_src,embed_tgt)
        scores.append(scores_pred)

    
    with open(os.path.join(os.path.dirname(data),'results_subtask2_en.tsv'),'w') as f:
        f.write('sim_context1\tsim_context2\n')

        for c1,c2 in scores:
            f.write('\t'.join([str(c1),str(c2)])+'\n')
    
    with open(os.path.join(os.path.dirname(data),'results_subtask1_en.tsv'),'w') as f:
        f.write('change\n')
        for c1,c2 in scores:
            score=c2-c1
            f.write(str(score)+'\n')

        