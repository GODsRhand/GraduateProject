from torch.cuda import amp
from torch.utils.data import DataLoader
from utils import vcfdf_to_hapsnp, generate_batch, Vocab, randomDataset
from model import BertForMLM
from model import BertConfig
from utils import logger_init
# from torch.utils.tensorboard import SummaryWriter
# from transformers import BertTokenizer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import argparse
import time
import torch
import os
import logging
import sys
sys.path.append('/home/linwenhao/Bert_imputation/Bert')


class ModelConfig:
    def __init__(self,
                 masked_rate=0.5,
                 test_masked_rate=0.5,
                 train_set="chr22_train_5000.csv",
                 test_set="chr22_test_5000.csv",
                 do_logging=True,
                 gpu="23",
                 learning_rate=1e-5,
                 batch_size=32,
                 epochs=750):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(self.project_dir, 'data', '1KGPMLM')
        self.pretrained_model_dir = os.path.join(
            self.project_dir, "bert_1KGPMLM")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        # self.device = torch.device(
        #     f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
        self.device = [torch.device(
            f'cuda:{int(x)}') for x in gpu] if torch.cuda.is_available() else [torch.device('cpu')]
        self.train_set = train_set
        self.test_set = test_set
        self.train_file_path = os.path.join(
            self.dataset_dir, train_set)
        self.val_file_path = os.path.join(
            self.dataset_dir, test_set)
        self.test_file_path = os.path.join(
            self.dataset_dir, test_set)
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        # self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.logs_save_dir = os.path.join(
            self.project_dir, '../records/bertmlm/raw')
        self.data_name = '1KGPMLM'
        self.model_save_path = os.path.join(
            self.model_save_dir, f'model_{self.data_name}.pt')
        self.is_sample_shuffle = True
        self.use_embedding_weight = True
        self.batch_size = batch_size
        self.max_sen_len = None  # 为None时则采用每个batch中最长的样本对该batch中的样本进行padding
        self.pad_index = 0
        self.random_state = 2021
        self.learning_rate = learning_rate
        self.masked_rate = masked_rate
        self.test_masked_rate = test_masked_rate
        self.masked_token_rate = 1
        self.masked_token_unchanged_rate = 1
        self.log_level = logging.DEBUG
        self.use_torch_multi_head = False  # False表示使用model/BasicBert/MyTransformer中的多头实现
        self.epochs = epochs
        self.model_val_per_epoch = 1
        self.do_logging = do_logging
        self.attention = ''

        if do_logging:
            logger_init(log_file_name='1KGP', log_level=self.log_level,
                        log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        bert_config_path = os.path.join(
            self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        if do_logging:
            logging.info(" ### 将当前配置打印到日志文件中 ")
            for key, value in self.__dict__.items():
                logging.info(f"### {key} = {value}")


def train(config):
    # 读取训练集数据
    start = time.time()
    voc = Vocab(config.vocab_path)
    end = time.time()
    print(os.path)
    if os.path.exists("./data/temp/train.npy"):
        train_nda = np.load(
            "./data/temp/train.npy", allow_pickle=True)
        print("the file has been processed")
    else:
        print("reading train file......")
        train_df = pd.read_csv(config.train_file_path, sep='\t')
        train_nda = vcfdf_to_hapsnp(train_df)
        np.save("./data/temp/train.npy", train_nda)
    end_f = time.time()
    print("read precocess_file time:", end_f-end)

    train_ds = randomDataset(train_nda,
                             hidden_size=config.hidden_size,
                             max_position_embeddings=config.max_position_embeddings,
                             masked_rate=config.masked_rate,
                             vocab=voc,
                             pad_index=voc['PAD'])
    end_rtime = time.time()
    train_iter = DataLoader(dataset=train_ds,
                            batch_size=config.batch_size,
                            collate_fn=generate_batch,
                            drop_last=True,
                            num_workers=8,
                            pin_memory=True)

    # 读取测试集数据
    if os.path.exists("./data/temp/test.npy"):
        test_nda = np.load("./data/temp/test.npy", allow_pickle=True)
        print("the file has been processed")
    else:
        print("reading test file......")
        test_df = pd.read_csv(config.test_file_path, sep='\t')
        test_nda = vcfdf_to_hapsnp(test_df)
        np.save("./data/temp/test.npy", test_nda)

    test_ds = randomDataset(test_nda,
                            hidden_size=config.hidden_size,
                            max_position_embeddings=config.max_position_embeddings,
                            masked_rate=config.masked_rate,
                            vocab=voc,
                            pad_index=voc['PAD'])
    val_iter = DataLoader(dataset=test_ds,
                          batch_size=config.batch_size,
                          collate_fn=generate_batch,
                          drop_last=True,
                          shuffle=True,
                          num_workers=8,
                          pin_memory=True)

    all_possible_position = torch.tensor(
        [train_ds.positions], dtype=torch.long)
    config.all_possible_position = all_possible_position  # slow

    model = BertForMLM(config)
    if os.path.exists(config.model_save_path):
        # loaded_paras = torch.load(config.model_save_path, map_location=torch.device(gpus))
        loaded_paras = torch.load(config.model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")

    model = model.to(config.device[0])
    model.to(config.device[0])
    model = nn.DataParallel(model, device_ids=config.device,
                            output_device=config.device[0])

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    max_acc = 0
    auc = 0

    # 精度加速
    scaler = amp.GradScaler()
    torch.backends.cuda.matmul.allow_tf32 = True

    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (b_token_ids, b_mask, b_mlm_label, b_positions) in enumerate(train_iter):
            b_token_ids = b_token_ids.to(config.device[0])
            b_mask = b_mask.to(config.device[0])
            b_mlm_label = b_mlm_label.to(config.device[0])
            b_positions = b_positions.to(config.device[0])
            model.train()
            optimizer.zero_grad()

            # 新增的代码
            with amp.autocast():
                loss, mlm_logits = model(input_ids=b_token_ids,
                                         attention_mask=b_mask,
                                         token_type_ids=None,
                                         position_ids=b_positions,
                                         masked_lm_labels=b_mlm_label,
                                         next_sentence_labels=None)

            # scaler.scale(loss).backward(torch.ones(loss.shape).to(gpus[0]))
            scaler.scale(loss).backward(
                torch.ones(loss.shape).to(config.device[0]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            losses += loss.sum().item()
            model.eval()
            mlm_acc, _, _ = accuracy(mlm_logits, b_mlm_label, voc['PAD'])
            matrix, f1, precision, recall, mcc = f1_precision_recall_mcc(
                mlm_logits, b_mlm_label, voc['PAD'])
            if idx % 20 == 0:
                logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.sum().item():.4f}, Train mlm acc: {mlm_acc:.4f}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, mcc: {mcc:.4f}")  # 一轮的结果

                # logging.info(f"\n{matrix[3:5, 3:5]}")
                logging.info(f"\n{matrix}")

        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: {epoch}, Train loss: "
                     f"{train_loss:.4f}, Epoch time = {(end_time - start_time):.4f}s")
        # attn_to_figure(attn_weight, epoch)
        if (epoch + 1) % config.model_val_per_epoch == 0:  # 整个epoch的结果
            mlm_acc, f1, precision, recall, mcc = evaluate(
                config, val_iter, model, voc['PAD'])
            auc = -1
            logging.info(
                f" ### MLM Accuracy on val: {round(mlm_acc, 4)}, roc_auc: {auc:.4f}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, mcc: {mcc}")
            if mlm_acc > max_acc:
                max_acc = mlm_acc
                torch.save(model.module.state_dict(), config.model_save_path)
                # torch.save(model, config.model_save_path)


def roc_auc(mlm_logits, mlm_labels, vocab_size, PAD_IDX):
    with torch.no_grad():
        mlm_pred = mlm_logits.reshape(-1, mlm_logits.shape[2])
        mlm_true = mlm_labels.reshape(-1)

        mask = torch.logical_not(mlm_true.eq(PAD_IDX))  # 获取mask位置的行索引
        mlm_pred = mlm_pred[mask, 3:]  # 去除预测为特殊标记可能性
        mlm_pred_sm = torch.softmax(mlm_pred, dim=1).cpu()
        mlm_true = mlm_true[mask]
        mlm_true = mlm_true.reshape(-1, 1).cpu()
        mlm_true = torch.zeros(mlm_true.shape[0], config.vocab_size).scatter_(   #
            dim=1, index=mlm_true, value=1)
        mlm_true = mlm_true[:, 3:]
        return roc_auc_score(y_true=mlm_true, y_score=mlm_pred_sm, average='macro', multi_class='ovr')


def f1_precision_recall_mcc(mlm_logits, mlm_labels, PAD_IDX):
    """
    :param mlm_logits:  [batch_size, src_len, vocab_size]
    :param mlm_labels:  [batch_size, src_len]
    :param PAD_IDX:
    :return:
    """
    mlm_pred = mlm_logits.argmax(axis=2).reshape(-1).cpu()
    mlm_true = mlm_labels.reshape(-1).cpu()
    mask = torch.logical_not(mlm_true.eq(PAD_IDX)).cpu().numpy()
    labels = list(range(mlm_logits.shape[-1]))
    matrix = confusion_matrix(
        mlm_true, mlm_pred, labels=labels, sample_weight=mask)
    TP = matrix[4][4]
    TN = matrix[3][3]
    FP = matrix[3][4]
    FN = matrix[4][3]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)
    mcc = matthews_corrcoef(mlm_true, mlm_pred, sample_weight=mask)

    return matrix, f1, precision, recall, mcc


def accuracy(mlm_logits, mlm_labels, PAD_IDX):
    """
    :param mlm_logits:  [batch_size, src_len, vocab_size]
    :param mlm_labels:  [batch_size, src_len]
    :param PAD_IDX:
    :return:
    """

    mlm_pred = mlm_logits.argmax(axis=2).reshape(-1)
    mlm_true = mlm_labels.reshape(-1)

    mlm_acc = mlm_pred.eq(mlm_true)  # 计算预测值与正确值比较的情况
    # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    mask = torch.logical_not(mlm_true.eq(PAD_IDX))
    mlm_acc = mlm_acc.logical_and(mask)  # 去掉acc中mask的部分
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    mlm_acc = float(mlm_correct) / mlm_total
    return (mlm_acc, mlm_correct, mlm_total)


def evaluate(config, data_iter, model, PAD_IDX, inference=False):
    model.eval()
    mlm_corrects, mlm_totals, auc, cnt = 0, 0, 0, 0
    for idx, (b_token_ids, b_mask, b_mlm_label, b_positions) in enumerate(data_iter):
        b_token_ids = b_token_ids.to(
            config.device[0])  # [src_len, batch_size]
        b_mask = b_mask.to(config.device[0])
        b_mlm_label = b_mlm_label.to(config.device[0])
        b_positions = b_positions.to(config.device[0])
        # 原来的代码
        a = time.time()
        _, mlm_logits = model(input_ids=b_token_ids,
                              attention_mask=b_mask,
                              token_type_ids=None,
                              position_ids=b_positions,
                              masked_lm_labels=b_mlm_label,
                              next_sentence_labels=None)
        b = time.time()

        # 加速：amp精度
        # with amp.autocast():
        #     _, mlm_logits, attn_weight = model(input_ids=b_token_ids,
        #                                 attention_mask=b_mask,
        #                                 token_type_ids=None,
        #                                 position_ids=b_positions,
        #                                 masked_lm_labels=b_mlm_label,
        #                                 next_sentence_labels=None)
        result = accuracy(mlm_logits, b_mlm_label, PAD_IDX)
        matrix, f1, precision, recall, mcc = f1_precision_recall_mcc(
            mlm_logits, b_mlm_label, PAD_IDX)

        if idx < 1:
            print(f"一次推断时间：{(b - a):.4f}s")
            mlm_pred = mlm_logits.transpose(
                0, 1).argmax(axis=2).transpose(0, 1)
            mask = torch.logical_not(b_mlm_label.eq(PAD_IDX))
            full_mask = torch.full_like(mlm_pred, fill_value=-1)
            mlm_result = torch.where(mask, mlm_pred, full_mask).tolist()
            mlm_to_pred = torch.where(mask, full_mask, b_token_ids).tolist()
            logging.debug(f" ## 待预测样本:{mlm_to_pred[0]}")
            logging.debug(f" ## 预测的结果:{mlm_result[0]}")
            print("\n")

        _, mlm_cor, mlm_tot = result
        mlm_corrects += mlm_cor
        mlm_totals += mlm_tot
        # auc += roc_auc(mlm_logits, b_mlm_label, config.vocab_size, PAD_IDX)
        cnt += 1
    model.train()
    return (float(mlm_corrects) / mlm_totals, f1, precision, recall, mcc)
    # return (float(mlm_corrects) / mlm_totals, auc / cnt, f1, precision, recall, mcc)


def inference(config):
    # 读取待推断数据
    voc = Vocab(config.vocab_path)

    time_start = time.time()
    if os.path.exists("./data/temp/test.npy"):
        val_nda = np.load(
            "./data/temp/train.npy", allow_pickle=True)
        print("the file has been processed")
    else:
        print("reading val file......")
        val_df = pd.read_csv(config.val_file_path, sep='\t')
        val_nda = vcfdf_to_hapsnp(val_df)
        np.save("./data/temp/test.npy", val_nda)

    val_ds = randomDataset(val_nda,
                           hidden_size=config.hidden_size,
                           max_position_embeddings=config.max_position_embeddings,
                           masked_rate=config.masked_rate,
                           vocab=voc,
                           pad_index=voc['PAD'])
    val_iter = DataLoader(dataset=val_ds,
                          batch_size=config.batch_size,
                          collate_fn=generate_batch)

    all_possible_position = torch.tensor([val_ds.positions], dtype=torch.long)
    config.all_possible_position = all_possible_position  # slow

    model = BertForMLM(config)
    if os.path.exists(config.model_save_path):
        loaded_paras = torch.load(config.model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行推断......")
    else:
        logging.info("## 已训练模型不存在，退出......")

    # model = model.to(config.device)
    model = model.to(config.device[0])
    time_load = time.time()
    mlm_acc, f1, precision, recall, mcc = evaluate(
        config, val_iter, model, voc['PAD'], inference=True)
    time_end = time.time()
    logging.info(
        f" ### MLM Accuracy on val: {round(mlm_acc, 4)}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, mcc: {mcc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="To do MLM task on 1KGP dataset.")
    parser.add_argument('-lr', '--learning_rate', type=float,
                        help='Masking rate of SNPs.', default=5e-5)
    parser.add_argument('-mr', '--masked_rate', type=float,
                        help='Masking rate of SNPs.', default=0.4)
    parser.add_argument('-tmr', '--test_masked_rate', type=float,
                        help='Masking rate of SNPs in testing set.', default=0.4)
    parser.add_argument('-i', '--inference', action='store_true',
                        help='To do inference on testing set with trained model.')
    parser.add_argument('-trs', '--train_set', type=str,
                        help='train_set', default="simplify.train.recode.vcf")
    parser.add_argument('-tes', '--test_set', type=str,
                        help='test_set', default="simplify.test.recode.vcf")
    parser.add_argument('-g', '--gpus', type=str,
                        help='gpus', default="23")
    parser.add_argument('-bsz', '--batch_size', type=int,
                        help='batch_size', default=64)
    parser.add_argument('-epo', '--epochs', type=int,
                        help='epochs', default=750)

    args = parser.parse_args()
    if args.gpus is None:
        args.gpus = "2"

    config = ModelConfig(masked_rate=args.masked_rate,
                         learning_rate=args.learning_rate,
                         test_masked_rate=args.test_masked_rate,
                         train_set=args.train_set,
                         test_set=args.test_set,
                         gpu=args.gpus,
                         batch_size=args.batch_size)

    if not args.inference:
        train(config)
    else:
        inference(config)
