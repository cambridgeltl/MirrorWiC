# MirrorWiC 
Code repo for the **CoNLL 2021** paper:

[_**MirrorWiC: On Eliciting Word-in-Context Representations from Pretrained Language Models**_](https://arxiv.org/abs/2109.09237)<br>
by [Qianchu Liu](https://qianchu.github.io/)\*, [Fangyu Liu](http://fangyuliu.me/about)\*, [Nigel Collier](https://sites.google.com/site/nhcollier/), [Anna Korhonen](https://sites.google.com/site/annakorhonen/), [Ivan Vulić](https://sites.google.com/site/ivanvulic/)

MirrorWiC is a fully unsupervised approach to improving word-in-context (WiC) representations in pretrained language models, achieved via a simple and efficient WiC-targeted fine-tuning procedure. The proposed method leverages only raw texts sampled from Wikipedia, assuming no sense-annotated data, and learns context-aware word representations within a standard contrastive learning setup.

## Huggingface Pretrained Models

|model | WiC (dev) | Usim |
|------|------|------|
|baseline: bert-base-uncased | 68.49 | 54.52 |
|[mirrorwic-bert-base-uncased](https://huggingface.co/cambridgeltl/mirrorwic-bert-base-uncased)| **71.94** | 61.82 |
|[mirrorwic-roberta-base](https://huggingface.co/cambridgeltl/mirrorwic-roberta-base)| 71.15 | 57.95 |
|[mirrorwic-deberta-base](https://huggingface.co/cambridgeltl/mirrorwic-deberta-base)| 71.78 | **62.79** |

## Train
1. Preprocess train data:
Run the following to convert a text file (one sentence per line) to WiC-formated train data. ``./train_data/en_wiki.txt`` provides example input. In the output data, each target word is marked with brackets and random erasing with masking is applied. 

```bash
>> python get_mirrorwic_traindata.py \
   --data [input data] \
   --lg [language] \
   --random_er [random erasing length]
```
Eg. 
```bash
>> python get_mirrorwic_traindata.py \
   --data ./train_data/en_wiki.txt \
   --lg en \
   --random_er 10
```
       
 2. Train:
   
```bash
>> cd train_scripts
>> bash ./mirror_sentence.sh [CUDA] [training data] [base model] [dropout]
```
Eg. 
```bash
>> bash ./mirror_wic.sh 1,0 ../train_data/en_wiki.txt.mirror.wic.re10 bert-base-uncased 0.4
```
    
## Evaluate
   
Download the evaluation data from [here](https://www.dropbox.com/s/c87cdj7l6ovq8nx/eval_data.zip?dl=0), and put the folder in the root directory. 

Then run: 
```bash     
>> cd evaluation_scripts
>> bash ./eval.sh [task] [model] [cuda]
```
`[task]`: `wic`, `wic-tsv`, `usim`, `cosimlex`, `wsd`, `am2ico`, `xlwic`
    
Eg. 
```bash
>> bash ./eval.sh usim cambridgeltl/mirrorwic-bert-base-uncased 0
```
   

## Citation
```bibtex
@inproceedings{liu2021mirrorwic,
  title={MirrorWiC: On Eliciting Word-in-Context Representations from Pretrained Language Models},
  author={Liu, Qianchu and Liu, Fangyu and Collier, Nigel and Korhonen, Anna and Vuli{\'c}, Ivan},
  booktitle = "Proceedings of the 25rd Conference on Computational Natural Language Learning (CoNLL)"
  year={2021}
}
```

## Acknolwedgement
The code is modified on the basis of [mirror-bert](https://github.com/cambridgeltl/mirror-bert).

      
