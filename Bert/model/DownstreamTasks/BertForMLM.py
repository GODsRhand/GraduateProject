import logging
from ..BasicBert.Bert import BertModel
from ..BasicBert.Bert import get_activation
from .BertForNSPAndMLM import BertForLMTransformHead
from .BertForNSPAndMLM import BertForMaskedLM
import torch.nn as nn


class BertForMLM(nn.Module):
    """
    BERT预训练模型，包括MLM任务
    """

    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForMLM, self).__init__()
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:  # 如果没有指定预训练模型路径，则随机初始化整个网络权重
            self.bert = BertModel(config)
        weights = None
        if 'use_embedding_weight' in config.__dict__ and config.use_embedding_weight:
            weights = self.bert.bert_embeddings.word_embeddings.embedding.weight
            logging.info(f"## 使用token embedding中的权重矩阵作为输出层的权重！{weights.shape}")
        self.mlm_prediction = BertForLMTransformHead(config, weights)
        self.config = config

    def forward(self, input_ids,  # [batch_size, src_len]
                attention_mask=None,  # [batch_size, src_len] mask掉padding部分的内容
                token_type_ids=None,  # [src_len, batch_size]
                position_ids=None,
                masked_lm_labels=None,  # [batch_size, src_len]
                next_sentence_labels=None):  # [batch_size]  
        # print("input_ids",input_ids.shape)
        
        # if masked_lm_labels is not None:
        # masked_lm_labels = masked_lm_labels.transpose(0, 1)
        attention_mask = attention_mask.transpose(0, 1)
        
        pooled_output, all_encoder_outputs, attn_weight = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)

  
      
        sequence_output = all_encoder_outputs[-1]  # 取Bert最后一层的输出
        # sequence_output: [src_len, batch_size, hidden_size]
        mlm_prediction_logits = self.mlm_prediction(sequence_output)
        # mlm_prediction_logits: [src_len, batch_size, vocab_size]
        mlm_prediction_logits = mlm_prediction_logits.transpose(0, 1)
        # mlm_prediction_logits: [batch_size, src_len, vocab_size]

        # print("masked_lm_labels",masked_lm_labels.shape)
        # print("mlm_prediction_logits",mlm_prediction_logits.shape)

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            mlm_loss = loss_fct(mlm_prediction_logits.reshape(-1, self.config.vocab_size),
                                masked_lm_labels.reshape(-1))
            mlm_loss = mlm_loss.unsqueeze(dim=0)
            # print("mlm_loss",mlm_loss.shape)
            return mlm_loss, mlm_prediction_logits
        else:
            return mlm_prediction_logits
        # [batch_size, src_len, vocab_size], [batch_size, 2]
