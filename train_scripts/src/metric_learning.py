import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
import random
from torch.cuda.amp import autocast
from pytorch_metric_learning import miners, losses, distances
from collections import defaultdict
LOGGER = logging.getLogger(__name__)


class Sap_Metric_Learning_pairwise(nn.Module):
    def __init__(self, encoder, learning_rate, weight_decay, use_cuda, 
            loss, use_miner=True, miner_margin=0.2, type_of_triplets="all", agg_mode="cls", infoNCE_tau="0.04"):

        LOGGER.info("Sap_Metric_Learning_pairwise! learning_rate={} weight_decay={} use_cuda={} loss={} infoNCE_tau={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
            learning_rate,weight_decay,use_cuda,loss,infoNCE_tau,use_miner,miner_margin,type_of_triplets,agg_mode
        ))
        super(Sap_Metric_Learning_pairwise, self).__init__()
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.optimizer = optim.AdamW([{'params': self.encoder.parameters()},], 
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        
        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5) # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(temperature=infoNCE_tau) # sentence: 0.04, word: 0.2, phrase: 0.04  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()
        elif self.loss == "contrastive_loss":
            self.loss = losses.ContrastiveLoss()
        elif self.loss == "two_term_loss":
            self.loss = losses.ContrastiveLoss(pos_margin=0.0,neg_margin=-10.0)

        print ("miner:", self.miner)
        print ("loss:", self.loss)
    
    @autocast() 
    def forward(self, query_toks1, query_toks2, labels):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """

        outputs1 = self.encoder(input_ids=query_toks1['input_ids'], attention_mask=query_toks1['attention_mask'],return_dict=True, output_hidden_states=True)
        outputs2 = self.encoder(input_ids=query_toks2['input_ids'], attention_mask=query_toks2['attention_mask'], return_dict=True, output_hidden_states=True)
        last_hidden_state1 = outputs1.last_hidden_state
        last_hidden_state2 = outputs2.last_hidden_state

        pooler_output1 = outputs1.pooler_output
        pooler_output2 = outputs2.pooler_output

        if self.agg_mode=="cls":
            query_embed1 = last_hidden_state1[:,0]  # query : [batch_size, hidden]
            query_embed2 = last_hidden_state2[:,0]  # query : [batch_size, hidden]
       

        elif self.agg_mode=='tokenmarker4layer':
            hidden_states1 = outputs1.hidden_states
            hidden_states2 = outputs2.hidden_states
            last_hidden_state1=sum(hidden_states1[-4:])/4
            last_hidden_state2=sum(hidden_states2[-4:])/4
            for num in range(last_hidden_state1.size()[0]):
                embeds_per_sent=last_hidden_state1[num]
                token_ids_per_sent=query_toks1['token_ids'][num]
                embed_token=torch.mean(embeds_per_sent[int(token_ids_per_sent[0]):int(token_ids_per_sent[1])],dim=0,keepdim=True)
                assert not torch.isnan(embed_token).any()
                if num == 0:
                    query_embed1 = embed_token
                else:
                    query_embed1 = torch.cat((query_embed1, embed_token),0)
            for num in range(last_hidden_state2.size()[0]):
                embeds_per_sent=last_hidden_state2[num]
                token_ids_per_sent=query_toks2['token_ids'][num]
                embed_token=torch.mean(embeds_per_sent[int(token_ids_per_sent[0]):int(token_ids_per_sent[1])],dim=0,keepdim=True)
                assert not torch.isnan(embed_token).any()
                if num == 0:
                    query_embed2 = embed_token
                else:
                    query_embed2 = torch.cat((query_embed2, embed_token),0)

        else:
            raise NotImplementedError()

        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        label2is=defaultdict(list)
        for i,label in enumerate(labels):
            label2is[label].append(i)
        query_embed1_avgtype=[]
        query_embed2_avgtype=[]
        for label in label2is:
            query_embed1_avgtype.append(query_embed1[label2is[label]].mean(0))
            query_embed2_avgtype.append(query_embed2[label2is[label]].mean(0))
        query_embed_type= torch.cat([torch.stack(query_embed1_avgtype,0), torch.stack(query_embed2_avgtype,0)], dim=0)

        labels_all_disambig=torch.tensor(list(range(len(labels))))
        labels_type=torch.tensor(list(range(len(label2is))))
        labels_all_disambig=torch.cat([labels_all_disambig, labels_all_disambig], dim=0)
        labels_type=torch.cat([labels_type,labels_type],dim=0)
        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            return self.loss(query_embed, labels, hard_pairs) 
        else:
            return (self.loss(query_embed_type, labels_type)+self.loss(query_embed,labels_all_disambig))/2


    def reshape_candidates_for_encoder(self, candidates):
        """
        reshape candidates for encoder input shape
        [batch_size, topk, max_length] => [batch_size*topk, max_length]
        """
        _, _, max_length = candidates.shape
        candidates = candidates.contiguous().view(-1, max_length)
        return candidates

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table



