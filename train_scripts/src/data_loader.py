import re
import os
import glob
import numpy as np
import random
import random
import pandas as pd
import json
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
#import spacy
#nlp = spacy.load("en_core_web_sm")
LOGGER = logging.getLogger(__name__)



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
        print (tokenizer.convert_ids_to_tokens(input_id))
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

class QueryDataset_custom(Dataset):

    def __init__(self, data_dir, 
                load_full_sentence=False,
                filter_duplicate=False
        ):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_duplicate={}".format(
            data_dir, filter_duplicate
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            filter_duplicate=filter_duplicate
        )
        
    def load_data(self, data_dir, filter_duplicate):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        with open(data_dir, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.rstrip("\n")
            _id, mention = line.split("||")
             
            data.append((mention, _id))

        if filter_duplicate:
            data = list(dict.fromkeys(data))
        
        # return np.array data
        data = np.array(data, dtype=object)
        
        return data

class QueryDataset_pretraining(Dataset):

    def __init__(self, data_dir, 
                filter_duplicate=False
        ):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_duplicate={}".format(
            data_dir,filter_duplicate
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            filter_duplicate=filter_duplicate
        )
        
    def load_data(self, data_dir, filter_duplicate):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        #concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
        with open(data_dir, "r") as f:
            lines = f.readlines()

        for row in lines:
            row = row.rstrip("\n")
            snomed_id, mention = row.split("||")
            data.append((mention, snomed_id))

        if filter_duplicate:
            data = list(dict.fromkeys(data))
        
        # return np.array data
        data = np.array(data)
        
        return data

class QueryDataset(Dataset):

    def __init__(self, data_dir, 
                filter_composite=False,
                filter_duplicate=False
        ):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries
        draft : bool
            use subset of queries for debugging (default False)     
        """
        LOGGER.info("QueryDataset! data_dir={} filter_composite={} filter_duplicate={}".format(
            data_dir, filter_composite, filter_duplicate
        ))
        
        self.data = self.load_data(
            data_dir=data_dir,
            filter_composite=filter_composite,
            filter_duplicate=filter_duplicate
        )
        
    def load_data(self, data_dir, filter_composite, filter_duplicate):
        """       
        Parameters
        ----------
        data_dir : str
            a path of data
        filter_composite : bool
            filter composite mentions
        filter_duplicate : bool
            filter duplicate queries  
        
        Returns
        -------
        data : np.array 
            mention, cui pairs
        """
        data = []

        #concept_files = glob.glob(os.path.join(data_dir, "*.txt"))
        file_types = ("*.concept", "*.txt")
        concept_files = []
        for ft in file_types:
            concept_files.extend(glob.glob(os.path.join(data_dir, ft)))

        for concept_file in tqdm(concept_files):
            with open(concept_file, "r", encoding='utf-8') as f:
                concepts = f.readlines()

            for concept in concepts:
                #print (concept)
                concept = concept.split("||")
                #if len(concept) !=5: continue
                mention = concept[3].strip().lower()
                cui = concept[4].strip()
                if cui.lower() =="cui-less": continue
                is_composite = (cui.replace("+","|").count("|") > 0)

                if filter_composite and is_composite:
                    continue
                else:
                    data.append((mention,cui))
        
        if filter_duplicate:
            data = list(dict.fromkeys(data))
        
        # return np.array data
        data = np.array(data)
        
        return data


class DictionaryDataset():
    """
    A class used to load dictionary data
    """
    def __init__(self, dictionary_path):
        """
        Parameters
        ----------
        dictionary_path : str
            The path of the dictionary
        draft : bool
            use only small subset
        """
        LOGGER.info("DictionaryDataset! dictionary_path={}".format(
            dictionary_path 
        ))
        self.data = self.load_data(dictionary_path)
        
    def load_data(self, dictionary_path):
        name_cui_map = {}
        data = []
        with open(dictionary_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip()
                if line == "": continue
                cui, name = line.split("||")
                name = name.lower()
                if cui.lower() == "cui-less": continue
                data.append((name,cui))
        
        #LOGGER.info("concerting loaded dictionary data to numpy array...")
        #data = np.array(data)
        return data


def erase_and_mask_token(s, erase_len=3):
    for i,token in enumerate(s):
        if token=='[':
           token_start=i
        elif token==']':
            token_end=i+1
    token_left=s[:token_start]
    token_right=s[token_end:]
    token=s[token_start:token_end]
    if random.sample([1,2],1)[0]==1:
        token_left=reorder(token_left,erase_len)
    else:
        token_right=reorder(token_right,erase_len)
    return token_left+' '+token+ ' '+token_right
    
def reorder(s,shuffle_len):
    wlist=s.split()
    if len(wlist) <= shuffle_len: return s
    ind = np.random.randint(len(wlist)-shuffle_len)
    select_shuffle=wlist[ind:ind+shuffle_len]
    random.shuffle(select_shuffle)
    wlist=wlist[:ind]+select_shuffle+wlist[ind+shuffle_len:]
    return ' '.join(wlist)

def erase_and_mask(s, erase_len=3):
    if len(s) <= erase_len: return s
    if len(s) < 5: return s # if too short, no augmentation
    ind = np.random.randint(len(s)-erase_len)
    left, right = s.split(s[ind:ind+erase_len], 1)
    return " ".join([left, "[MASK]", right])

class MetricLearningDataset_pairwise(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, path, tokenizer, random_erase=0): #d_ratio, s_score_matrix, s_candidate_idxs):
        with open(path, 'r') as f:
            lines = f.readlines()
        self.query_ids = []
        self.query_names = []
        for line in lines:
            line = line.rstrip("\n")
            query_id, name1, name2 = line.split("||")
            self.query_ids.append(query_id)
            self.query_names.append((name1, name2))
        self.tokenizer = tokenizer
        self.query_id_2_index_id = {k: v for v, k in enumerate(list(set(self.query_ids)))}
        self.random_erase = random_erase
    
    def __getitem__(self, query_idx):

        query_name1 = self.query_names[query_idx][0]
        query_name2 = self.query_names[query_idx][1]
        if self.random_erase != 0:
            query_name2 = erase_and_mask_token(query_name2, erase_len=int(self.random_erase))
            # print (query_name2)
        query_id_orig = self.query_ids[query_idx]
        query_id = int(self.query_id_2_index_id[query_id_orig])

        return query_name1, query_name2, query_id, query_id_orig


    def __len__(self):
        return len(self.query_names)



