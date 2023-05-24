import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', "--source", type=str, help="源文件所在目录")
parser.add_argument('--gen', type=str, default="/home/linwenhao/raw_data/dealer/gencsv.py", help="gencsv.py所在位置")
parser.add_argument('-p', "--prefix", type=str, default="out", help="待生成csv文件的前缀")
parser.add_argument('-o', "--out", type=str, default="./", help="输出目录")
parser.add_argument('--temp', type=str, default="./", help="缓存文件夹")
args = parser.parse_args()

files = os.listdir(args.source)
for file in files:
    if file.startswith(args.prefix) and file.endswith(".vcf") and not file.endswith("recode.vcf"):
        file_real_prefix = file[:-4]
        # print(file_real_prefix)
        file_path = os.path.join(args.source, file)
        out_path = os.path.join(args.out, file_real_prefix)
        os.system(f"python {args.gen} --file {file_path} --out {out_path}")