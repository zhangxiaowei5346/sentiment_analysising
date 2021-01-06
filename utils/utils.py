import torch
from torchtext import data
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot


# 计算开销时间
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - elapsed_mins * 60)
    return elapsed_mins, elapsed_secs


# 读取participle()函数保存的json文件，并使用torchtext加载
def load_json(TEXT, LABEL, train_file_name):
    fields = {'text': ('t', TEXT), 'label': ('l', LABEL)}
    train_data = data.TabularDataset.splits(
        path='data/',
        train=train_file_name,
        format='json',
        fields=fields,
    )
    return train_data[0]


# 划分数据集，随机取80%正类+80%负类作为训练集，其余作为验证集
def dataset_division(train_data_all, proportion, SEED):
    # train_data size: 4283
    # positive number: 1908  negative number: 2375

    train_data, valid_data = train_data_all.split(random_state=random.seed(SEED),
                                                  split_ratio=proportion)

    return train_data, valid_data


# 统计需要求导的变量总数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#
def binary_accuracy(preds, y):
    # torch.round 返回每个输入元素最近的整数
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def plot_acc_loss(loss, acc):
    host = host_subplot(111)
    plt.subplots_adjust(right=0.8)
    par1 = host.twinx()

    host.set_xlabel("steps")
    host.set_ylabel("validation-loss")
    host.set_ylabel("validation-accuracy")

    p1, = host.plot(range(len(loss)), loss, label="loss")
    p2, = par1.plot(range(len(acc)), acc, label="accuracy")

    host.legend(loc=5)

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    plt.draw()
    plt.show()

