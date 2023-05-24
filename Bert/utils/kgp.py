import math
import os
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time

def vcfdf_to_hapsnp(df: pd.DataFrame, ridx: pd.Index=None, is_impute: bool=False, temp="./temp/vcfdf_to_hapsnp_temp") -> tuple:
    """
    将读入vcf的dataframe中的基因型的列数据转化为单倍型的列数据
    """
    ids = df.columns
    # print(ids)
    if is_impute:
        df = df.reindex(columns=ridx)
        df.to_csv(temp, sep='\t', index=False)
        df = pd.read_csv(temp, sep='\t')
        ids = df.columns
    # 从第9列开始是位点信息
    
    for id in ids[9:]: 
        df[str(id) + "_0"] = df[id].apply(lambda x: int(x[0]))
        df[str(id) + "_1"] = df[id].apply(lambda x: int(x[-1]))
    
    # print(df)
    
    # 删除原有信息
    df.drop(columns=ids[5:], inplace=True)
    df.drop(columns=['#CHROM', 'ID'], inplace=True)
    df['POS'] = pd.to_numeric(df['POS'])
    nda = df.to_numpy().T
    # print(df['POS'].dtype)
    # print(nda)
    return nda

class Vocab:
    """
    根据本地的vocab文件，构造一个词表
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    print(len(vocab))  # 返回词表长度
    """
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors

def generate_batch(data_batch, PAD_IDX=0, max_len=512):
    b_token_ids, b_mlm_label, b_positions = [], [], []
    for (token_ids, mlm_label, positions) in data_batch:
        a = time.time()
        # 开始对一个batch中的每一个样本进行处理
        b_token_ids.append(token_ids)
        b_mlm_label.append(mlm_label)
        b_positions.append(positions)
        b = time.time()

    b_token_ids = pad_sequence(b_token_ids,  # [batch_size,max_len]
                               padding_value=PAD_IDX,
                               batch_first=False,
                               max_len=max_len)
    b_token_ids = b_token_ids.transpose(0,1)                        
    #b_token_ids:  [batch_size,src_len]
    
    
    b_mlm_label = pad_sequence(b_mlm_label,  # [batch_size,max_len]
                               padding_value=PAD_IDX,
                               batch_first=False,
                               max_len=max_len)
   # b_mlm_label:  [src_len, batch_size]
   
    b_mask = (b_token_ids == PAD_IDX).transpose(0, 1)
   # b_mask: [max_len, batch_size]
   
    b_mlm_label = b_mlm_label.transpose(0, 1)
    b_mask = b_mask.transpose(0, 1)
    # b_mlm_label:  [batch_size, src_len]
    # b_mask: [batch_size, max_len]
   
    # b_positions = pad_sequence(b_positions,  # [batch_size, max_len, hidden_size]
    #                            padding_value=PAD_IDX,
    #                            batch_first=False,
    #                            max_len=1)
    
    # print(f"b_positions.type: {type(b_positions)}")
    b_positions = torch.stack(b_positions)
    # print(f"b_positions.shape: {b_positions.shape}")
    #b_positions:  [batch_size, max_len, hidden_size]

    return b_token_ids, b_mask, b_mlm_label, b_positions


class randomDataset(Dataset):
    def __init__(self, 
                 nda:np.ndarray, 
                 hidden_size:int,
                 max_position_embeddings:int=512, 
                 masked_rate=0.4, 
                 vocab:Vocab=None, 
                 pad_index=0,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5,
                 ) -> None:
        super().__init__()
        self.positions = nda[0]
        self.positions_tensor = torch.from_numpy(self.positions.astype(np.int_)).reshape(-1,1)
        self.ref = nda[1]
        self.alt = nda[2]
        self.data = nda[3:]
        self.samples = self.data.shape[0]
        self.length = self.data.shape[1]
        self.max_position_embeddings = max_position_embeddings
        self.max_start = self.length - max_position_embeddings
        self.vocab = vocab
        self.masked_rate = masked_rate
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        # print(f"position dtype: {self.positions.astype(np.int).dtype}")
        # print(f"positions shape: {self.positions_tensor.shape}")
        # print(f"data shape: {self.data.shape}")
        
        # 预先计算正余弦编码
        self.pe = torch.zeros(self.length, hidden_size)
        self.div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(self.positions_tensor * self.div_term)  # 用正弦波给偶数部分赋值
        self.pe[:, 1::2] = torch.cos(self.positions_tensor * self.div_term)  # 用余弦波给奇数部分赋值

        # print(self.data[0][0], self.vocab[str(self.data[0][0])])
    
    def get_masked_sample(self, token_ids):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        masked_rate = self.masked_rate
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        # random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，BERT模型中默认将15%的Token进行mask
        num_mlm_preds = max(1, round(len(token_ids) * masked_rate))
        # logging.debug(f" ## Mask数量为: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        # print(mlm_input_tokens_id.shape)
        return mlm_input_tokens_id, mlm_label
        
    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        """
        根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """
        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            masked_token_id = None
            # 80%的时间：将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            if random.random() < self.masked_rate:
                masked_token_id = self.MASK_IDS
                mlm_input_tokens_id[mlm_pred_position] = masked_token_id
                pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id
        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        return mlm_input_tokens_id, mlm_label
    
    def __getitem__(self, index):
        # 转化为token_id
        
        # 取随机起始位点的索引
        rand_start = random.randint(0, self.max_start)
       
        token_ids = self.data[index][rand_start: rand_start + self.max_position_embeddings]
        token_ids = [self.vocab[str(token)] for token in token_ids]
        self.data[index][rand_start:rand_start + self.max_position_embeddings]
        position_embedding = self.pe[rand_start: rand_start + self.max_position_embeddings]
        # print(type(token_ids))
        mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids)
        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
        mlm_label = torch.tensor(mlm_label, dtype=torch.long)
        # print(f"in randomset: {mlm_label.shape}")
        rand_start = torch.tensor(rand_start, dtype=torch.long).unsqueeze(0)
        # positions = torch.tensor(list(positions), dtype=torch.long)
        
        return token_ids, mlm_label, position_embedding
        
    def __len__(self):
        return self.samples
    
class maskedDataset(randomDataset):
    def __init__(self, 
                 nda: np.ndarray,
                 answer_nda: np.ndarray, 
                 hidden_size: int, 
                 max_position_embeddings: int = 512, 
                 masked_rate=0.4, 
                 vocab: Vocab = None, 
                 pad_index=0, 
                 masked_token_rate=0.8, 
                 masked_token_unchanged_rate=0.5) -> None:
        super().__init__(nda, hidden_size, max_position_embeddings, masked_rate,
                         vocab, pad_index, masked_token_rate, masked_token_unchanged_rate)
        self.answer_data = answer_nda[3:]
        
    def get_masked_sample_for_test(self, token_ids, answer_ids):
        candidate_pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            if token_ids[i] == self.MASK_IDS:
                mlm_input_tokens_id[i] = self.MASK_IDS
                candidate_pred_positions.append(i)
        mlm_label = [self.PAD_IDX if idx not in candidate_pred_positions else answer_ids[idx]
                     for idx in range(len(token_ids))]
        # print(mlm_input_tokens_id.shape)
        return mlm_input_tokens_id, mlm_label
    
    def __getitem__(self, index):
        # 转化为token_id

        # 取随机起始位点的索引
        rand_start = random.randint(0, self.max_start)

        token_ids = self.data[index][rand_start: rand_start + self.max_position_embeddings]
        token_ids = [self.vocab[str(token)] for token in token_ids]
        answer_ids = self.answer_data[index][rand_start:rand_start + self.max_position_embeddings]
        answer_ids = [self.vocab[str(token)] for token in answer_ids]
        position_embedding = self.pe[rand_start: rand_start + self.max_position_embeddings]
        # print(type(token_ids))
        mlm_input_tokens_id, mlm_label = self.get_masked_sample_for_test(token_ids, answer_ids)
        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
        mlm_label = torch.tensor(mlm_label, dtype=torch.long)
        # print(f"in randomset: {mlm_label.shape}")
        rand_start = torch.tensor(rand_start, dtype=torch.long).unsqueeze(0)
        # positions = torch.tensor(list(positions), dtype=torch.long)

        return token_ids, mlm_label, position_embedding