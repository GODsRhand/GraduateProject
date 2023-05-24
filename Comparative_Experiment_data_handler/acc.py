import os
import pandas as pd
import numpy as np
import argparse
from tools import remove_front, add_front, is_vcf_format

def vcfdf_to_hapsnp(df: pd.DataFrame, ridx: pd.Index=None, is_impute: bool=False, temp="./output/try.vcf") -> tuple:
    """
    将读入vcf的dataframe中的基因型的列数据转化为单倍型的列数据
    """
    ids = df.columns
    print(ids)
    if is_impute:
        df = df.reindex(columns=ridx)
        df.to_csv(temp, sep='\t', index=False)
        df = pd.read_csv(temp, sep='\t')
        ids = df.columns
    # 从第9列开始是位点信息
    for id in ids[9:]: 
        df[str(id) + "_0"] = df[id].apply(lambda x: x[0])
        df[str(id) + "_1"] = df[id].apply(lambda x: x[-1])
    # 删除原有信息
    df.drop(columns=ids, inplace=True)
    nda = df.to_numpy()
    return (nda, (ids if not is_impute else None))

def accuracy(answer, masked, impute, tempdf):
    if is_vcf_format(answer):
        remove_front(answer, "./temp_answer")
        answer = "./temp_answer"
    if is_vcf_format(masked):
        remove_front(masked, "./temp_masked")
        masked = "./temp_masked"
    if is_vcf_format(impute):
        remove_front(impute, "./temp_impute")
        impute = "./temp_impute"
    
    df_masked = pd.read_csv(masked, sep='\t')
    df_answer = pd.read_csv(answer, sep='\t')
    df_impute = pd.read_csv(impute, sep='\t')

    np_masked, ids = vcfdf_to_hapsnp(df_masked)
    np_answer, _ = vcfdf_to_hapsnp(df_answer)
    np_impute, _ = vcfdf_to_hapsnp(df_impute, ridx=ids, is_impute=True, temp=tempdf)
    np_summary = np.dstack((np_masked, np_answer, np_impute)).transpose(2, 0, 1)
    print(np_summary.shape)

    masked_idx = np_masked == '.'
    masked_cnt = np_masked[masked_idx].size
    cor_idx = (np_answer == np_impute) & masked_idx
    cor_cnt = np_impute[cor_idx].size

    print(f"masked_cnt : {masked_cnt}, cor_cnt = {cor_cnt}, acc = {cor_cnt / masked_cnt}")
    return masked_cnt, cor_cnt, cor_cnt / masked_cnt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', "--answer_path", type=str, default="./chr22_test_1024.vcf")
    parser.add_argument('-m', "--masked_path", type=str, default="./masked.vcf")
    parser.add_argument('-i', "--impute_path", type=str, default="./chr22_5e-2_imp.recode.vcf")
    parser.add_argument('-t', "--tempdf_path", type=str, default="./output/try.vcf")
    
    args = parser.parse_args()
    answer, masked, impute, tempdf = args.answer_path, args.masked_path, args.impute_path, args.tempdf_path

    accuracy(answer, masked, impute, tempdf)

    
    

    
