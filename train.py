import argparse
from tqdm import tqdm
from loguru import logger

import numpy as np
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset import TrainDataset, TestDataset
from model import SimcseModel, simcse_unsup_loss, simcse_sup_loss
from transformers import BertModel, BertConfig, BertTokenizer
import os
from os.path import join
from torch.utils.tensorboard import SummaryWriter
import random
import pickle
import pandas as pd
import time


def seed_everything(seed=42):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def train(model, train_loader, dev_loader, optimizer, args):
    logger.info("start training")
    model.train()
    device = args.device
    best = 0
    for epoch in range(args.epochs):
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx
            # [batch, n, seq_len] -> [batch * n, sql_len]
            sql_len = data['input_ids'].shape[-1]
            input_ids = data['input_ids'].view(-1, sql_len).to(device)
            attention_mask = data['attention_mask'].view(-1, sql_len).to(device)
            token_type_ids = data['token_type_ids'].view(-1, sql_len).to(device)

            out = model(input_ids, attention_mask, token_type_ids)
            if args.train_mode == 'unsupervise':
                loss = simcse_unsup_loss(out, device)
            else:
                loss = simcse_sup_loss(out, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % args.eval_step == 0:
                corrcoef = evaluate(model, dev_loader, device)
                logger.info('loss:{}, corrcoef: {} in step {} epoch {}'.format(loss, corrcoef, step, epoch))
                writer.add_scalar('loss', loss, step)
                writer.add_scalar('corrcoef', corrcoef, step)
                model.train()
                if best < corrcoef:
                    best = corrcoef
                    torch.save(model.state_dict(), join(args.output_path, 'simcse.pt'))
                    logger.info('higher corrcoef: {} in step {} epoch {}, save model'.format(best, step, epoch))


def evaluate(model, dataloader, device):
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in tqdm(dataloader):
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(device)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(device)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(device)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(device)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(device)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(device)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def load_train_data_unsupervised(tokenizer, args):
    """
    获取无监督训练语料
    """
    logger.info('loading unsupervised train data')
    output_path = os.path.dirname(args.output_path)
    train_file_cache = join(output_path, 'train-unsupervise.pkl')
    if os.path.exists(train_file_cache) and not args.overwrite_cache:
        with open(train_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of train data:{}".format(len(feature_list)))
            return feature_list
    feature_list = []
    with open(args.train_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        # lines = lines[:100]
        logger.info("len of train data:{}".format(len(lines)))
        for line in tqdm(lines):
            line = line.strip()
            feature = tokenizer([line, line], max_length=args.max_len, truncation=True, padding='max_length', return_tensors='pt')
            feature_list.append(feature)
    with open(train_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return feature_list


def load_train_data_supervised(tokenizer, args):
    """
    获取NLI监督训练语料
    """
    logger.info('loading supervised train data')
    output_path = os.path.dirname(args.output_path)
    train_file_cache = join(output_path, 'train-supervised.pkl')
    if os.path.exists(train_file_cache) and not args.overwrite_cache:
        with open(train_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of train data:{}".format(len(feature_list)))
            return feature_list
    feature_list = []
    df = pd.read_csv(args.train_file, sep=',')
    logger.info("len of train data:{}".format(len(df)))
    rows = df.to_dict('records')
    # rows = rows[:10000]
    for row in tqdm(rows):
        sent0 = row['sent0']
        sent1 = row['sent1']
        hard_neg = row['hard_neg']
        feature = tokenizer([sent0, sent1, hard_neg], max_length=args.max_len, truncation=True, padding='max_length', return_tensors='pt')
        feature_list.append(feature)
    with open(train_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return feature_list


def load_eval_data(tokenizer, args, mode):
    """
    加载验证集或者测试集
    """
    assert mode in ['dev', 'test'], 'mode should in ["dev", "test"]'
    logger.info('loading {} data'.format(mode))
    output_path = os.path.dirname(args.output_path)
    eval_file_cache = join(output_path, '{}.pkl'.format(mode))
    if os.path.exists(eval_file_cache) and not args.overwrite_cache:
        with open(eval_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of {} data:{}".format(mode, len(feature_list)))
            return feature_list

    if mode == 'dev':
        eval_file = args.dev_file
    else:
        eval_file = args.test_file
    feature_list = []
    with open(eval_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        logger.info("len of {} data:{}".format(mode, len(lines)))
        for line in tqdm(lines):
            line = line.strip().split("\t")
            assert len(line) == 7 or len(line) == 9
            score = float(line[4])
            data1 = tokenizer(line[5].strip(), max_length=args.max_len, truncation=True, padding='max_length', return_tensors='pt')
            data2 = tokenizer(line[6].strip(), max_length=args.max_len, truncation=True, padding='max_length', return_tensors='pt')

            feature_list.append((data1, data2, score))
    with open(eval_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return feature_list


def main(args):
    # 加载模型
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    assert args.pooler in ['cls', "pooler", "last-avg", "first-last-avg"], \
        'pooler should in ["cls", "pooler", "last-avg", "first-last-avg"]'
    model = SimcseModel(pretrained_model=args.pretrain_model_path, pooling=args.pooler, dropout=args.dropout).to(
        args.device)
    if args.do_train:
        # 加载数据集
        assert args.train_mode in ['supervise', 'unsupervise'], \
            "train_mode should in ['supervise', 'unsupervise']"
        if args.train_mode == 'supervise':
            train_data = load_train_data_supervised(tokenizer, args)
        elif args.train_mode == 'unsupervise':
            train_data = load_train_data_unsupervised(tokenizer, args)
        train_dataset = TrainDataset(train_data, tokenizer, max_len=args.max_len)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True,
                                      num_workers=args.num_workers)
        dev_data = load_eval_data(tokenizer, args, 'dev')
        dev_dataset = TestDataset(dev_data, tokenizer, max_len=args.max_len)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=True,
                                    num_workers=args.num_workers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train(model, train_dataloader, dev_dataloader, optimizer, args)
    if args.do_predict:
        test_data = load_eval_data(tokenizer, args, 'test')
        test_dataset = TestDataset(test_data, tokenizer, max_len=args.max_len)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=True,
                                     num_workers=args.num_workers)
        model.load_state_dict(torch.load(join(args.output_path, 'simcse.pt')))
        model.eval()
        corrcoef = evaluate(model, test_dataloader, args.device)
        logger.info('testset corrcoef:{}'.format(corrcoef))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="gpu or cpu")
    parser.add_argument("--output_path", type=str, default='output')
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size_train", type=int, default=4)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_step", type=int, default=100, help="every eval_step to evaluate model")
    parser.add_argument("--max_len", type=int, default=64, help="max length of input")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--train_file", type=str, default="data/nli_for_simcse.csv")
    parser.add_argument("--dev_file", type=str, default="data/stsbenchmark/sts-dev.csv")
    parser.add_argument("--test_file", type=str, default="data/stsbenchmark/sts-test.csv")
    parser.add_argument("--pretrain_model_path", type=str,
                        default="pretrain_model/bert-base-uncased")
    parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg"],
                        default='cls', help='pooler to use')
    parser.add_argument("--train_mode", type=str, default='unsupervise', choices=['unsupervise', 'supervise'], help="unsupervise or supervise")
    parser.add_argument("--overwrite_cache", action='store_true', default=False, help="overwrite cache")
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_predict", action='store_true', default=True)

    args = parser.parse_args()
    seed_everything(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")
    args.output_path = join(args.output_path, args.train_mode, 'bsz-{}-lr-{}-dropout-{}'.format(args.batch_size_train, args.lr, args.dropout))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
    logger.info(args)
    writer = SummaryWriter(args.output_path)
    main(args)


