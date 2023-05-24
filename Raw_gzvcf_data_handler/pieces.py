import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', "--source", type=str, help="源文件")
parser.add_argument('-l', "--len", type=int, default=512, help="分片长度")
parser.add_argument('-p', "--prefix", type=str, default="out", help="输出前缀")
parser.add_argument('-o', "--out", type=str, default="./", help="输出目录")
parser.add_argument('--temp', type=str, default="./", help="缓存文件夹")
args = parser.parse_args()

prefix = os.path.join(args.out, args.prefix)

with open(file=args.source, mode="r") as src:
    # 找到表头所在的行(start_line)
    while(True):
        line = src.readline()
        if line.startswith("#CHROM"):
            start_line = line
            break

    cnt = 0 # 片段计数器
    
    # 收集长度为args.len的片段，末尾不足args.len的片段抛弃
    while(True):
        lines = [start_line]
        flag = True
        for _ in range(args.len):
            line = src.readline()
            if not line:
                flag = False # 如果不足args.len行，标记为非
                break
            lines.append(line)
        if not flag:
            break
        # 符合条件
        output = prefix + f".{cnt}.vcf"
        with open(file=output, mode="w+") as vcf:
            vcf.writelines(lines)
        cnt = cnt + 1
                