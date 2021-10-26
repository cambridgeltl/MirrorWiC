import numpy as np
import sys
import torch
import nltk 

NORMALIZE='normalize'
CENTER='center'


def produce_orig_to_tok_map(orig_tokens,tokenizer):
    ### Input

    ### Output
    bert_tokens = []

    # Token map will be an int -> int mapping between the `orig_tokens` index and
    # the `bert_tokens` index.
    orig_to_tok_map = []

    bert_tokens.append("[CLS]")
    for orig_token in orig_tokens:
        orig_to_tok_map.append(len(bert_tokens))
        bert_tokens.extend(tokenizer.tokenize(orig_token,add_special_tokens=False))
    bert_tokens.append("[SEP]")
    return bert_tokens,orig_to_tok_map



# bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
# orig_to_tok_map == [1, 2, 4, 6]
def matrix_norm(emb):
    emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb))

def produce_cos_matrix(test_src,test_tgt):
    normalize_embeddings(test_src, NORMALIZE, None)
    normalize_embeddings(test_tgt, NORMALIZE, None)
    cos_matrix = torch.mm(test_src, test_tgt.transpose(0, 1))
    return cos_matrix

def normalize_embeddings(emb, types, mean=None):
    """
    Normalize embeddings by their norms / recenter them.
    """
    for t in types.split(','):
        if t == '':
            continue
        if t == CENTER:
            if mean is None:
                mean = emb.mean(0, keepdim=True)
            emb.sub_(mean.expand_as(emb))
        elif t == NORMALIZE:
            matrix_norm(emb)
        else:
            raise Exception('Unknown normalization type: "%s"' % t)
    return mean if mean is not None else None

def produce_cosine_list(test_src,test_tgt):
    cos_matrix=produce_cos_matrix(test_src, test_tgt)
    scores_pred = [float(cos_matrix[i][i]) for i in range(len(cos_matrix))]
    return scores_pred

def delete_tokenmark_input(input_ids,tokenizer):
    input_id_new=[]
    del_num=0
    token_pos_start_id=[tokenizer.encode('[',add_special_tokens=False)[0],tokenizer.encode(' [',add_special_tokens=False)[0]]
    token_pos_end_id=[tokenizer.encode(']',add_special_tokens=False)[0],tokenizer.encode(' ]',add_special_tokens=False)[0]]
    token_pos_start_end_id=set(token_pos_start_id+token_pos_end_id)
    for i,input_i in enumerate(input_ids):
        if input_i not in token_pos_start_end_id:
            input_id_new.append(input_i)
        else:
            del_num+=1
    input_id_new+=del_num*[tokenizer.pad_token_id]
    return input_id_new

def delete_tokenmarker_am(input_ids,tokenizer):
    am_new=[]
    for i in input_ids:
        if i==tokenizer.pad_token_id:
            am_new.append(0)
        else:
            am_new.append(1)
    return am_new

def find_token_id(input_id,tokenizer):
    token_pos_start_id=set([tokenizer.encode('[',add_special_tokens=False)[0],tokenizer.encode(' [',add_special_tokens=False)[0]])    
    token_pos_end_id=set([tokenizer.encode(']',add_special_tokens=False)[0],tokenizer.encode(' ]',add_special_tokens=False)[0]])    
    
    token_ids=[]
    for i,input_i in enumerate(input_id):
        input_i=int(input_i)
        if i==len(input_id)-1: # the last token
            continue
        if input_i in [tokenizer.mask_token_id,tokenizer.cls_token_id,tokenizer.pad_token_id]:
            continue
        if input_i in token_pos_start_id:
            token_ids.append(i+1)
            # logger.info("first word",token_ids)
        elif input_i in token_pos_end_id:
            token_ids.append(i)
    try:
        assert len(token_ids)==2
    except AssertionError as e:
        print ('Warning, token id alter is not length 2')
        print (input_id)
        print (tokenizer.convert_ids_to_tokens(input_id))
        print (token_pos_start_id)
        print (token_pos_end_id)
        print (token_ids)
        sys.exit(1)
   
    try:
       assert token_ids[1]!=token_ids[0]
    except AssertionError as e:
        print ('token marker star == end')
        print (input_id)
        print (token_ids)
        sys.exit(1)
    token_ids[1]=token_ids[1]-1
    token_ids[0]=token_ids[0]-1
    return token_ids
    
def delete_tokenmaker_tokentypeids(input_ids,tokenizer):
    tokentype_ids=[]
    item=0
    for i in input_ids:
    
        if i==tokenizer.pad_token_id:
            tokentype_ids.append(0)
        
        elif i==tokenizer.sep_token_id:
            tokentype_ids.append(item)
            item=1
        else:
            tokentype_ids.append(item)  
    return tokentype_ids

def get_embed(sentences,tokenizer,model,flag='cls',layer_start=None,layer_end=None,maxlen=64):
    if flag=='cls':
        sentences=[sentence.replace('[','').replace(']','') for sentence in sentences]
        toks = tokenizer.batch_encode_plus(sentences, max_length = maxlen,truncation = True, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs_ = model(input_ids=toks['input_ids'].cuda(),attention_mask=toks['attention_mask'].cuda(), output_hidden_states=True)
        last_hidden_state = outputs_.last_hidden_state
        output = last_hidden_state.detach().cpu().numpy()[:,0]
    elif flag=='cls_with_token':
        toks = tokenizer.batch_encode_plus(sentences, max_length = maxlen,truncation = True, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs_ = model(input_ids=toks['input_ids'].cuda(),attention_mask=toks['attention_mask'].cuda(), output_hidden_states=True)
        last_hidden_state = outputs_.last_hidden_state
        output = last_hidden_state.detach().cpu().numpy()[:,0]
    elif flag=='mean':
        sentences=[sentence.replace('[','').replace(']','') for sentence in sentences]
        toks = tokenizer.batch_encode_plus(sentences, max_length = maxlen,truncation = True, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs_ = model(input_ids=toks['input_ids'].cuda(),attention_mask=toks['attention_mask'].cuda(), output_hidden_states=True)
        hidden_states = outputs_.hidden_states
        average_layer_batch = sum(hidden_states[layer_start:layer_end]) / (layer_end-layer_start)
        
        output = average_layer_batch.detach().cpu().numpy().mean(1)

    elif flag=='preappend':
        sentences=[sentence.split()[sentence.split().index('[')+1]+' $ '+ sentence for sentence in sentences]
        # print (sentences)
        toks = tokenizer.batch_encode_plus(sentences, max_length = maxlen,truncation = True, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs_ = model(input_ids=toks['input_ids'].cuda(),attention_mask=toks['attention_mask'].cuda(), output_hidden_states=True)
        last_hidden_state = outputs_.last_hidden_state
        output = last_hidden_state.detach().cpu().numpy()[:,0,:]
    elif flag=='alltoken':
        toks = tokenizer.batch_encode_plus(sentences, max_length = maxlen,truncation = True, padding="max_length", return_tensors="pt")

       
        # for num in range(average_layer_batch.size()[0]):
        #     embeds_per_sent=average_layer_batch[num]
        #     token_ids_per_sent=all_token_ids[num]
            
        #     embed_token=torch.mean(embeds_per_sent[int(token_ids_per_sent[0]):int(token_ids_per_sent[1])],dim=0,keepdim=True)
        #     # assert int(token_ids_per_sent[0])!=int(token_ids_per_sent[1])
        #     assert not torch.isnan(embed_token).any()
        #     if num == 0:
        #         output = embed_token
        #     else:
        #         output = torch.cat((output, embed_token),0)
        output = output.detach().cpu().numpy()
    elif flag.startswith('token'):
        toks = tokenizer.batch_encode_plus(sentences, max_length = maxlen,truncation = True, padding="max_length")
        all_token_ids=torch.tensor([find_token_id(tok,tokenizer) for tok in toks['input_ids']], dtype=torch.long).cuda()
        all_input_ids=torch.tensor([delete_tokenmark_input(tok,tokenizer) for tok in toks['input_ids']], dtype=torch.long).cuda()
        all_attention_mask=torch.tensor([delete_tokenmarker_am(input_ids,tokenizer) for input_ids in all_input_ids], dtype=torch.long).cuda()
        all_token_type_ids=torch.tensor([delete_tokenmaker_tokentypeids(input_ids,tokenizer) for input_ids in all_input_ids], dtype=torch.long).cuda()
        inputs = {"input_ids": all_input_ids, "attention_mask": all_attention_mask}
        with torch.no_grad():
            outputs_ = model(**inputs, output_hidden_states=True)
        hidden_states = outputs_.hidden_states
        average_layer_batch = sum(hidden_states[layer_start:layer_end]) / (layer_end-layer_start)
        
        for num in range(average_layer_batch.size()[0]):
            embeds_per_sent=average_layer_batch[num]
            token_ids_per_sent=all_token_ids[num]
            
            embed_token=torch.mean(embeds_per_sent[int(token_ids_per_sent[0]):int(token_ids_per_sent[1])],dim=0,keepdim=True)
            # assert int(token_ids_per_sent[0])!=int(token_ids_per_sent[1])
            assert not torch.isnan(embed_token).any()
            if num == 0:
                output = embed_token
            else:
                output = torch.cat((output, embed_token),0)
        output = output.detach().cpu().numpy()
        if flag=='token+cls':
            last_hidden_state = outputs_.last_hidden_state
            output=np.concatenate([output, last_hidden_state.detach().cpu().numpy()[:,0]],axis=1)
            # print (output.shape)
        
    return output


nltk.download('punkt')
def lg2wordtokenize(lg):
    if lg =='en':
        from nltk.tokenize import word_tokenize, sent_tokenize
    elif lg in ['sw','ht','kk','ml','te','tl','ko','ta','ka','ms','es','id','ar','fi','pl','fr','it','de','eu']:
        from nltk.tokenize import wordpunct_tokenize as word_tokenize
        from nltk.tokenize import sent_tokenize
    elif lg=='bn':
        
        from bnlp import BasicTokenizer
        from bnlp import NLTKTokenizer 
        bnltk = NLTKTokenizer()
        def sent_tokenize(text):
            return bnltk.sentence_tokenize(text)  
        def word_tokenize(text):
            return bnltk.word_tokenize(text)  
    elif lg=='ur':
        from urdu import _generate_sentences as sent_tokenize
       
        def word_tokenize(text):
            return text.split()
    elif lg=='th':
        from pythainlp.tokenize import word_tokenize as wt
        def sent_tokenize(text):
            sent_lst=text.split('\n')
            return sent_lst
        def word_tokenize(text):
            return ' '.join(wt(text, engine="newmm")).split()
    elif lg=='my':
        def sent_tokenize(text):
            sent_lst=text.split('\n')
            return sent_lst
        def word_tokenize(text):
            return text.split()
    elif lg=='tr':
        from sentence_splitter import SentenceSplitter, split_text_into_sentences
        def sent_tokenize(text):
            splitter = SentenceSplitter(language=lg)
            sent_lst=splitter.split(text)
            return sent_lst
        from nltk.tokenize import wordpunct_tokenize as word_tokenize
    elif lg=='ru':
        from razdel import sentenize
        def sent_tokenize(text):
            sent_list=[sent.text for sent in list(sentenize(text))]
            return sent_list
        from nltk.tokenize import wordpunct_tokenize as word_tokenize
        
    elif lg =='zh':
        import opencc
        import re
        converter = opencc.OpenCC('t2s.json')
        import jieba
        def word_tokenize(text):
            text=' '.join(text.split())
            seg_list = [w for w in jieba.cut(text, cut_all=False) if w.strip()]
            return seg_list
        def sent_tokenize(text):
            sent_list=[text]
            return sent_list

    elif lg=='ja':
        import nagisa
        import re
        def word_tokenize(text):
            words = nagisa.tagging(text)
            try:
                wordslist=' '.join(words.words).split()
            except AssertionError as e:
                print(e)
                print ('assertion error', 264)
                return None
            return wordslist
        def sent_tokenize(text):
            sent_list=re.findall(u'[^!?。\.\!\?]+[!?。\.\!\?]?', text, flags=re.U)
            return sent_list
    return word_tokenize,sent_tokenize