# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        print("\nClassifier i/p shape: ", x.shape)
        x = x.reshape(-1,x.size(-1)*2)
        print("\nClassifier i/p(after reshape) shape: ", x.shape)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
    
        
    def forward(self, input_ids=None,labels=None): 
        input_ids=input_ids.view(-1,self.args.block_size)
        print("\nEncoder i/p shape: ", input_ids.shape)
        outputs = self.encoder(input_ids= input_ids,attention_mask=input_ids.ne(1))[0]
        print("\nEncoder o/p shape: ", outputs.shape)
        logits=self.classifier(outputs)
        print("\nClassifier o/p shape: ", logits.shape)
        prob=F.softmax(logits)
        print("\nProbabilities shape: ", prob.shape)
        print("\nLabels: ", labels)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            print("\nLoss func: ", loss_fct)
            loss = loss_fct(logits, labels)
            print("\nLoss: ", loss)
            return loss,prob
        else:
            return prob
      
        
 
        

