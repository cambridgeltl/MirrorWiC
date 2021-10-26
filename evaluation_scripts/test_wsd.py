

if __name__ =='__main__':

    import sys
    from src.helper import * 
    from collections import defaultdict
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm.auto import tqdm
    import nltk
    nltk.download('wordnet')
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import wordpunct_tokenize
    lemmatizer = WordNetLemmatizer()
    import os
    bsz = 128
    maxlen = 100 # 64
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

    sid2src=defaultdict(list)
    sid2tgt=defaultdict(list)
    sid2golds=defaultdict(list)
    sid2examples=defaultdict(list)
    for line in tqdm(open(data).readlines()[1:]):
        sid, label, sent, gloss,start,end,key=line.strip().split('\t')
        examples=[wordpunct_tokenize(example) for example in wn.lemma_from_key(key).synset().examples()]

        if flag.startswith('token'):
            wlst=sent.split()
            w=wlst[int(start):int(end)]
            wlemma=lemmatizer.lemmatize(' '.join(w))
            wlst.insert(int(start),'[')
            wlst.insert(int(end)+1,']')
            sent=' '.join(wlst)
            gloss='[ '+' '.join(w)+' ] means '+gloss
            examples_new=[]

            for example in examples:
                for i,w in enumerate(example):
                    if lemmatizer.lemmatize(w)==wlemma:
                        example.insert(i,'[')
                        example.insert(i+2,']')
                        examples_new.append(' '.join(example))
                        break
            examples=examples_new
        else:
            examples=[' '.join(example) for example in examples]
        sid2examples[sid].append(examples)
        sid2tgt[sid].append((gloss,key))
        sid2src[sid].append(sent)
        if label=='1':
            sid2golds[sid].append(key)

    precision=0
    with open(data+'.preds.'+flag+'.'+os.path.basename(model_name),'w') as f:
        for sid in tqdm(sid2src):
            src_sent=list(set(sid2src[sid]))
            assert len(src_sent)==1
            tgt_sents,tgt_keys=list(zip(*sid2tgt[sid]))
            tgt_examples=sid2examples[sid]
            tgt_sents,src_sent=list(tgt_sents),list(src_sent)
            embed_src=get_embed(src_sent,tokenizer,model,flag,layer_start,layer_end,maxlen)
            embeds_examples_candidates=[]
            for i,examples_per_sense in enumerate(tgt_examples):
                examples_per_sense.append(tgt_sents[i])
                embeds_examples=get_embed(examples_per_sense,tokenizer,model,flag,layer_start,layer_end,maxlen)
                embed_examples=np.mean(embeds_examples,0)
                embeds_examples_candidates.append(embed_examples)
            embed_tgt=np.stack(embeds_examples_candidates,0)
            assert len(embed_tgt)==len(tgt_sents)
            scores_pred=cosine_similarity(embed_src,embed_tgt)[0]
            tgtid=np.argmax(scores_pred)
            key_pred=tgt_keys[tgtid]
            golds=sid2golds[sid]
            f.write(sid+' '+key_pred+'\n')
            if key_pred in golds:
                precision+=1
    print ('precision',precision/len(sid2golds))

    
        