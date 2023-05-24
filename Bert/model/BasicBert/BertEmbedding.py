import torch.nn as nn
import torch
from torch.nn.init import normal_
import pandas as pd
import math, time


class PositionalEmbedding(nn.Module):
    
    def __init__(self, hidden_size, batch_size, max_position_embeddings=512, initializer_range=0.02, all_possible_position=None, device=None):
        super(PositionalEmbedding, self).__init__()
        assert max_position_embeddings >= 512, "config.max_position_embeddings参数必须大于等于512"
        # 因为BERT预训练模型的长度为512
        self.device = device
        self._reset_parameters(initializer_range)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_position_embeddings = max_position_embeddings
        self.pre_dense = torch.empty((self.batch_size, self.max_position_embeddings, self.hidden_size))
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
        self.simple_pe = nn.Linear(1, hidden_size, bias=True)
        self.activation = nn.Sigmoid()


    def forward(self, position_ids):
        """
        :param position_ids: [position_ids_len, batch_size]
        :return: [position_ids_len, batch_size, hidden_size]
        """

        result = self.activation(self.dense(position_ids)).transpose(0, 1)
        return result


    def _reset_parameters(self, initializer_range):
        r"""Initiate parameters."""
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token_id=0, initializer_range=0.02):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self._reset_parameters(initializer_range)

    def forward(self, input_ids):
        """
        :param input_ids: shape : [input_ids_len, batch_size]
        :return: shape: [input_ids_len, batch_size, hidden_size]
        """
        return self.embedding(input_ids)

    def _reset_parameters(self, initializer_range):
        r"""Initiate parameters."""
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class SegmentEmbedding(nn.Module):
    def __init__(self, type_vocab_size, hidden_size, initializer_range=0.02):
        super(SegmentEmbedding, self).__init__()
        self.embedding = nn.Embedding(type_vocab_size, hidden_size)
        self._reset_parameters(initializer_range)

    def forward(self, token_type_ids):
        """

        :param token_type_ids:  shape: [token_type_ids_len, batch_size]
        :return: shape: [token_type_ids_len, batch_size, hidden_size]
        """
        return self.embedding(token_type_ids)

    def _reset_parameters(self, initializer_range):
        r"""Initiate parameters."""
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class BertEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.word_embeddings = TokenEmbedding(vocab_size=config.vocab_size,
                                              hidden_size=config.hidden_size,
                                              pad_token_id=config.pad_token_id,
                                              initializer_range=config.initializer_range)
        # return shape [src_len,batch_size,hidden_size]

        self.position_embeddings = PositionalEmbedding(max_position_embeddings=config.max_position_embeddings,
                                                       hidden_size=config.hidden_size,
                                                       batch_size=config.batch_size,
                                                       initializer_range=config.initializer_range,
                                                       all_possible_position=config.all_possible_position,
                                                       device=config.device)
        # return shape [src_len,1,hidden_size]

        self.token_type_embeddings = SegmentEmbedding(type_vocab_size=config.type_vocab_size,
                                                      hidden_size=config.hidden_size,
                                                      initializer_range=config.initializer_range)
        # return shape  [src_len,batch_size,hidden_size]

        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids",
                             torch.arange(config.max_position_embeddings).expand((1, -1)))
        # shape: [1, max_position_embeddings]

    def forward(self,
                input_ids=None,
                position_ids=None,
                token_type_ids=None):
        """
        :param input_ids: [src_len, batch_size]
        :param position_ids: [1,src_len]
        :param token_type_ids: [src_len,batch_size]
        :return: [src_len, batch_size, hidden_size]
        """
        src_len = input_ids.size(0)
        token_embedding = self.word_embeddings(input_ids).transpose(0,1)
        # shape:[src_len,batch_size,hidden_size]
        # print(f"token_embedding devices: {token_embedding.device}")

        if position_ids is None:
            position_ids = self.position_ids[:, :src_len]  # [1,src_len]
        positional_embedding = self.position_embeddings(position_ids)
        # [src_len, 1, hidden_size]

        embeddings = token_embedding + positional_embedding
        # [src_len,batch_size,hidden_size] + [src_len,1,hidden_size] + [src_len,batch_size,hidden_size]

        embeddings = self.LayerNorm(embeddings)  # [src_len, batch_size, hidden_size]
        #modify
        embeddings = embeddings.transpose(0,1)
        embeddings = self.dropout(embeddings)
        # print("embedding shape",embeddings.shape)
        return embeddings
