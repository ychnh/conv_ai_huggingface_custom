import os
import math
import logging

from collections import defaultdict
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from apex import amp
from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer

from optim import AdamW


SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]


##########################################################
# TRAIN ##################################################
##########################################################

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    num_added_tokens = tokenizer.set_special_tokens(SPECIAL_TOKENS)
    model.set_num_special_tokens(len(SPECIAL_TOKENS))


def train():

    EPOCHS = 3
    SAVE_ITR = 3
    LM_COEF = 1.0
    MC_COEF = 1.0
    DEVICE = 0
    FP16 = True
    MAX_NORM = 1.0 # Clipping Gradient Norm
    GRAD_ACCUM_STEPS = 6#4
    train_batch_size = 3#4

    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')

    add_special_tokens_(model, tokenizer)

    model = model.cuda(DEVICE)
    optimizer = AdamW(model.parameters(), lr=6.25e-5, correct_bias=True)
    if FP16: 
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1') #O1/O2 #https://nvidia.github.io/apex/amp.html

    train_dataset = torch.load('train_dataset.pyobj')
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)


    def update(b, batch):
        model.train()

        batch = [input_tensor.to(DEVICE) for input_tensor in batch]
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        lm_loss, mc_loss = model(input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids) 
        loss = (lm_loss * LM_COEF + mc_loss * MC_COEF) / GRAD_ACCUM_STEPS

        if FP16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), MAX_NORM)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)

        if b % GRAD_ACCUM_STEPS == 0: optimizer.step(); optimizer.zero_grad()
        return loss.item()

    E,B = EPOCHS, len(train_loader)
    for e in range(EPOCHS):
        for b,batch in enumerate(train_loader):

            loss = update(b,batch)
            if b%(B//300) == 0: 
                print(e,str(b)+'/'+str(B), loss )

            torch.cuda.empty_cache()

        if (e+1)%SAVE_ITR==0: torch.save( model.state_dict(), '/media/sec/conv_ai_weights/'+str(e+1)+'.pth')

train()

##########################################################
# DATA PREP ##############################################
##########################################################

NUM_CANDIDATES = 2 # Cap on number of Train Candidates
PERSONALITY_PERM = 1 # Number of permutations of personality sentences
MXHST_K = 2
MAX_HISTORY = 2*(MXHST_K)+1 # Number of previous exchanges to keep in history

def get_data_loaders(tokenizer):
    personachat = torch.load('convai_data.tkn')

    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        
        #num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        num_candidates =  20 #MAXIMUM NUM OF DISTRACTOR+GT_REPLY in our dataset
        if dataset_name == 'train': num_candidates = min(NUM_CANDIDATES, num_candidates) # Number of candidates for training
        datasets[dataset_name]["n_candidates"] = num_candidates
        
        for dialog in dataset: #dialog= [ personality:[], utterances:[history:[], candidates:[]] ]
            persona = dialog["personality"].copy()
            #for _ in range(PERSONALITY_PERM):  ----------------------------------------------
            for utterance in dialog["utterances"]:
                history = utterance["history"][-MAX_HISTORY:] #MAX_HISTORY per person
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    lm_labels = bool(j == num_candidates-1)
                    instance = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels)
                    #for input_name, input_array in instance.items(): #datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]['input_ids'].append( instance['input_ids'] )
                    datasets[dataset_name]['token_type_ids'].append( instance['token_type_ids'] )
                    datasets[dataset_name]['mc_token_ids'].append( instance['mc_token_ids'] )
                    datasets[dataset_name]['lm_labels'].append( instance['lm_labels'] )
                    
                datasets[dataset_name]["mc_labels"].append(num_candidates-1) #TODO: make this 0
                
            #persona = [persona[-1]] + persona[:-1]  #permuted personalities
            # PERSONALITY_PERM LOOP ----------------------------------------------------------
            
    personachat = None; del personachat
    #dataset['train'/'valid'] = {'input_ids', 'lm_labels', 'token_type_ids', 'mc_token_ids'}
    # The dataset contains lists that are grouped N=num_candidates objects
    
    #input_ids: sequence of token ids
    #lm_labels: sequence of token ids with highlisted reply (lang modeling)
    #token_type_ids: speaker annotation for each token
    #mc_token_ids: length of input id-1 (some index that indicates when padding starts)
    #mc_labels: index of the ground truth candidate (Multiple choice)
    
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        
        #dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids('<pad>') ) ---------------------------------
        pad = tokenizer.convert_tokens_to_ids('<pad>')
        dataset['input_ids'] = pad_list( dataset['input_ids'], pad)
        dataset['token_type_ids'] = pad_list( dataset['token_type_ids'], pad)
        dataset['lm_labels'] = pad_list( dataset['lm_labels'], -1)
        #-------------------------------------------------------------------------------------------------------------------
        
        for input_name in ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                
                #-----------------------------------------------------------------------------------------------------------
                N,L = datasets[dataset_name]["n_candidates"], tensor.shape[1:]
                tensor = tensor.view((-1, N) + L) 
                #L = tensor.shape[-1]; tensor = tensor.view((-1, N, L)) # Simpler version ----------------------------------
                
            tensor_datasets[dataset_name].append(tensor)

    train_dataset = TensorDataset(*tensor_datasets["train"])
    torch.save(train_dataset, 'train_dataset.pyobj')
    #valid_dataset = TensorDataset(*tensor_datasets["valid"])
    return train_dataset#, valid_loader#, train_sampler, valid_sampler

def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    else:
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        
    return instance

def pad_list(LIST, pad):
    L = max(len(x) for x in LIST ) 
    return [ x + (L-len(x))*[pad] for x in LIST ]

def test_dataset(dataset):
    d = dataset[0]
    input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = d
