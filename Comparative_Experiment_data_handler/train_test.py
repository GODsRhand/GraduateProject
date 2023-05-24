import os
import time
import logging
import argparse
from acc import accuracy
from tools import add_front, is_vcf_format

parser = argparse.ArgumentParser()
# 训练集(refPanel)路径
parser.add_argument("-tr", "--train", type=str)
# 测试集(Answer)路径
parser.add_argument("-te", "--test", type=str)
# 掩蔽率(maskRate)
parser.add_argument("-mr", "--mask_rate", type=float, default=0.4)
args = parser.parse_args()

# random_mask, 输出文件为/home/linwenhao/CE/data/mask.vcf
random_mask_out = "/home/linwenhao/CE/data/mask.vcf"
os.system(f"python /home/linwenhao/CE/handler/random_mask.py --file {args.test} --out {random_mask_out}")

# 确保训练集满足m3的格式要求
if is_vcf_format(args.train):
    ref_panel = args.train
else:
    ref_panel = "/home/linwenhao/CE/temp/temp_ref"
    add_front(args.train, ref_panel)

# 补全, 输出文件为/home/linwenhao/CE/output/impute.dose.vcf
impute_out = "/home/linwenhao/CE/output/impute"
os.system(f"/home/linwenhao/CE/Minimac3/bin/Minimac3-omp --refHaps {ref_panel} \
                            --haps {random_mask_out} \
                            --prefix {impute_out} \
                            --format GT \
                            --nobgzip \
                            --cpus 5")
impute_out = "/home/linwenhao/CE/output/impute.dose.vcf"

# 统计, 结果保留在~/records/minimac3, acc.py产生的缓存在~/CE/temp/temp_df
temp_df = "/home/linwenhao/CE/temp/temp_df"
masked_cnt, cor_cnt, acc = accuracy(answer=args.test, masked=random_mask_out, impute=impute_out, tempdf=temp_df)
time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
logging.basicConfig(filename=f'/home/linwenhao/records/minimac3/{time_str}', level=logging.DEBUG)
logging.info(f"referrence panel: {args.train}")
logging.info(f"test set: {args.test}")
logging.info(f"acc: {acc}")