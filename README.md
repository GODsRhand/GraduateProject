# README

## BERT Haplotype MLM Imputation

哈尔滨工业大学（深圳）计算机科学与技术学院19级本科毕业设计（林文昊，基于BERT的基因型补全方法）

### Quick Start

1. 克隆本项目: `git clone https://github.com/GODsRhand/GraduateProject.git`
2. 卸载原依赖: `pip uninstall -y -r requirement.txt`
3. 配置新依赖: `pip install -r requirements.txt`
4. 模型训练：
   - 需要设置训练的掩蔽率`masked_rate`,训练集文件`train_set`，测试集文件`test_set`
   - 可使用多个GPU并行计算，如使用0,1,2号GPU`--gpus 012`
   - 数据文件请放在`./data/1KGPMLM`下
5. 只用于补全任务时，请设置`--inference` 

### Environment

- Python 3.8

See more in `requirement.txt`.

### Repo Structure

```
root
├── Bert                                        // 主要项目代码
│   ├── bert_1KGPMLM
│   ├── cache
│   ├── data
│   ├── model
│   ├── TaskFor1KGPMLM.py                       // 主文件
│   └── utils
├── Comparative_Experiment_data_handler         // 使用 Minimac4 进行对照实验时的数据处理脚本
│   ├── acc.py                                  // 统计结果
│   ├── add_front.py
│   ├── chrom.py
│   ├── divide.py
│   ├── random_mask.py                          // 掩蔽
│   ├── remove.py
│   ├── sites.py
│   ├── tools
│   └── train_test.py
└── Raw_gzvcf_data_handler                      // 处理原始 VCF 数据时的脚本
    ├── density.py                              // 降低 snp 位点密度
    ├── gencsv_from_pieces.py
    ├── gencsv.py
    ├── pieces.py
    └── r2_anal.py
```

### License

This project is licensed under the terms of the MIT license.