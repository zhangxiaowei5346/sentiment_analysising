import numpy as np
import torch.optim as optim
import time
from model.RNN import *
from model.LSTM import *
from utils.utils import *
from model.BERTGRU import *
import argparse
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')


def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="data/project3_train.csv", help="data directory")
    args.add_argument("-epoch", "--epoch_num", type=int,
                      default=15, help="Number of epochs")
    args.add_argument("-pre_emb", "--pretrained_emb",
                      default=None, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=bool,
                      default=100, help="Size of embeddings")
    args.add_argument("-l", "--learning_rate", type=float, default=1e-3)
    args.add_argument("-g_best_model", "--get_best_model",
                      type=bool, default=True)
    args.add_argument("-u_best_model", "--use_best_model",
                      type=bool, default=False)
    args.add_argument("-model", "--model_choose", default='RNN',
                      help="choose the model")
    args.add_argument("-n_l", "--n_layers", type=int,
                      default=1, help="RNN num of layers")
    args.add_argument("-b_d", "--bidirectional", type=bool,
                      default=False)
    args.add_argument("-d_o", "--drop_out", type=float,
                      default=0)
    args.add_argument("-h_d", "--hidden_dim", type=int,
                      default=256)
    args.add_argument("-o_d", "--output_dim", type=int,
                      default=1)
    args.add_argument("-j_path", "--json_path",
                      default="project3_train.json")
    args.add_argument("-in_len", "--include_length", type=bool,
                      default=False)
    args.add_argument("-b_s", "--batch_size", type=int,
                      default=64)
    args.add_argument("-s_w_b", "--sort_within_batch", type=bool,
                      default=False)
    args = args.parse_args()
    return args


def pre_process():                          # BERT-GRU
    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token

    print(init_token, eos_token, pad_token, unk_token)

    init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
    eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
    pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
    unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

    print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

    TEXT = data.Field(batch_first=True,
                      use_vocab=False,
                      tokenize=tokenize_and_cut,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      init_token=init_token_idx,
                      eos_token=eos_token_idx,
                      pad_token=pad_token_idx,
                      unk_token=unk_token_idx)

    LABEL = data.LabelField(dtype=torch.float)

    fields = [('t', TEXT), ('l', LABEL)]
    train_data = data.TabularDataset.splits(
        path='data/',
        train='project3_train.csv',
        format='csv',
        fields=fields,
        skip_header=True
    )
    train_data = train_data[0]

    return TEXT, LABEL, train_data


args = parse_args()
print(args)
SEED = 1324
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('your device is : ', device)


def tokenize_and_cut(sentence):

    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:120]
    return tokens


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        if args.model_choose == 'LSTM':
            text, text_lengths = batch.t
            predictions = model(text, text_lengths).squeeze(1)
        elif args.model_choose == 'RNN':
            predictions = model(batch.t).squeeze(1)
        else:                                               # BERT+GRU
            predictions = model(batch.t).squeeze(1)
        loss = criterion(predictions, batch.l)
        acc = binary_accuracy(predictions, batch.l)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            if args.model_choose == 'LSTM':
                text, text_lengths = batch.t
                predictions = model(text, text_lengths).squeeze(1)
            elif args.model_choose == 'RNN':
                predictions = model(batch.t).squeeze(1)
            else:                                           # GRU
                predictions = model(batch.t).squeeze(1)
            loss = criterion(predictions, batch.l)
            acc = binary_accuracy(predictions, batch.l)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# 建立模型
def main():
    if args.model_choose == 'GRU':
        TEXT, LABEL, train_data_all = pre_process()
        train_data, valid_data = dataset_division(train_data_all, 0.8, SEED)
    else:
        TEXT = data.Field(include_lengths=args.include_length)  # LSTM True RNN False
        LABEL = data.LabelField(dtype=torch.float)
        train_data_all = load_json(TEXT, LABEL, args.json_path)
        train_data, valid_data = dataset_division(train_data_all, 0.8, SEED)      # train_data 3426 valid_data 857
        # RNN vocab build
        if args.model_choose == 'RNN':
            TEXT.build_vocab(train_data)
        elif args.model_choose == 'LSTM':
            TEXT.build_vocab(train_data, vectors=args.pretrained_emb, unk_init=torch.Tensor.normal_)  # 6476
        INPUT_DIM = len(TEXT.vocab)

    LABEL.build_vocab(train_data)  # 2

    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=args.batch_size,
        device=device,
        sort_key=lambda x: len(x.t),
        sort_within_batch=args.sort_within_batch
    )

    if args.model_choose == 'RNN':
        model = RNN(INPUT_DIM, args.embedding_size, args.hidden_dim, args.output_dim)  # initial model
    elif args.model_choose == 'LSTM':
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        model = LSTM(INPUT_DIM, args.embedding_size, args.hidden_dim, args.output_dim,
                     args.n_layers, args.bidirectional, args.drop_out, PAD_IDX)
        pretrained_embedding = TEXT.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embedding)
        model.embedding.weight.data[UNK_IDX] = torch.zeros(args.embedding_size)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(args.embedding_size)
    else:
        bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        model = BERTGRU(bert,
                        args.hidden_dim,
                        args.output_dim,
                        args.n_layers,
                        args.bidirectional,
                        args.drop_out)

        for name, param in model.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False
    print(f'The model has', count_parameters(model), 'trainable parameters')

    # train the model
    optimizer = optim.Adam(model.parameters())
    # BCEWithLogitsLoss() reference https://blog.csdn.net/qq_22210253/article/details/85222093
    criterion = nn.BCEWithLogitsLoss()  # 二进制交叉熵损失函数

    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float('inf')
    for epoch in range(args.epoch_num):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


if __name__ == '__main__':
    main()




