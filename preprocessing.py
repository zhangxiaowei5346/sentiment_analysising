import hanlp
import json
import torch
from torchtext import data
import argparse


def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="project3_train.csv", help="data directory(默认在data文件夹下)")
    args.add_argument("-j_s", "--json_out",
                      default="data/project3_train.json", help="data directory(默认在data文件夹下)")
    args = args.parse_args()
    return args


# 读取初始数据并进行分词，保存到json文件方便读取
def participle(train_file, output_file):
    tokenizer = hanlp.load('LARGE_ALBERT_BASE')
    TEXT = data.Field(tokenize=tokenizer)
    LABEL = data.LabelField(dtype=torch.float)
    fields = [('text', TEXT), ('label', LABEL)]
    train_data = data.TabularDataset.splits(
        path='data/',
        train=train_file,
        format='csv',
        fields=fields,
        skip_header=True
    )
    train_data = train_data[0]
    print("splits over")
    json_file = open(output_file, 'a', encoding='utf-8')
    print("ready to write")
    for index, text in enumerate(train_data):
        print(index, vars(text))
        data_json = json.dumps(vars(text), ensure_ascii=False)
        json_file.write(data_json + '\n')
        print("write successfully")
    json_file.close()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    participle(args.data, args.json_out)
