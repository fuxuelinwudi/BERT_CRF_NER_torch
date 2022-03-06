# -*- coding: utf-8 -*-

import os
import random
import numpy as np

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from src.utils.crf_utils.crf import CRF


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(2022)


class BERT_CRF(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BERT_CRF, self).__init__(config)

        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels + 2)
        self.crf = CRF(self.num_labels)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                labels=None):

        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss, predict = self.crf.neg_log_likelihood_loss(logits, attention_mask, labels), \
                            self.crf.decode(logits, attention_mask)

            return loss, predict

        else:
            predict = self.crf.decode(logits, attention_mask)

            return predict
