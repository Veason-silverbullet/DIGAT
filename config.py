import os
import argparse
import time
import torch
import random
import numpy as np
from prepare_MIND_dataset import prepare_MIND_small, prepare_MIND_large
from MIND_corpus import MIND_Corpus
import torch.distributed as dist
import datetime


class Config:
    def parse_argument(self):
        parser = argparse.ArgumentParser(description='DIGAT Experiments')
        # General config
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test'], help='Mode')
        parser.add_argument('--news_encoder', type=str, default='MSA', choices=['MSA', 'CNN'], help='News encoder')
        parser.add_argument('--graph_encoder', type=str, default='DIGAT', choices=['DIGAT', 'wo_SA', 'Seq_SA', 'wo_interaction', 'news_graph_wo_inter', 'user_graph_wo_inter'], help='Graph encoder')
        parser.add_argument('--dev_model_path', type=str, default='best_model/MIND-small/MSA-DIGAT/#1/MSA-DIGAT', help='Dev model path')
        parser.add_argument('--test_model_path', type=str, default='best_model/MIND-small/MSA-DIGAT/#1/MSA-DIGAT', help='Test model path')
        parser.add_argument('--test_output_file', type=str, default='', help='Test output file')
        parser.add_argument('--device_id', type=int, default=0, help='Device ID of GPU')
        parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
        parser.add_argument('--local_rank', type=int, default=-1, help='Local GPU rank for distributed training (-1 for single GPU)')
        # Dataset config
        parser.add_argument('--dataset', type=str, default='MIND-small', choices=['MIND-small', 'MIND-large'], help='Directory root of dataset')
        parser.add_argument('--word_threshold', type=int, default=3, help='Word threshold')
        parser.add_argument('--max_title_length', type=int, default=32, help='Sentence truncate length for title')
        # Training config
        parser.add_argument('--negative_sample_num', type=int, default=4, help='Negative sample number of each positive sample')
        parser.add_argument('--max_history_num', type=int, default=50, help='Maximum number of history news for each user')
        parser.add_argument('--epoch', type=int, default=16, help='Training epoch')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
        parser.add_argument('--gradient_clip_norm', type=float, default=1, help='Gradient clip norm (non-positive value for no gradient clipping)')
        # Dev config
        parser.add_argument('--dev_criterion', type=str, default='avg', choices=['auc', 'mrr', 'ndcg5', 'ndcg10', 'avg'], help='Dev criterion to select model')
        parser.add_argument('--early_stopping_epoch', type=int, default=5, help='Epoch of stop training after the dev result does not improve')
        # Model config
        parser.add_argument('--word_embedding_dim', type=int, default=300, choices=[50, 100, 200, 300], help='Word embedding dimension')
        parser.add_argument('--cnn_method', type=str, default='naive', choices=['naive', 'group3', 'group4', 'group5'], help='CNN group')
        parser.add_argument('--cnn_kernel_num', type=int, default=400, help='Number of CNN kernel')
        parser.add_argument('--cnn_window_size', type=int, default=3, help='Window size of CNN kernel')
        parser.add_argument('--MSA_head_num', type=int, default=16, help='Head number of multihead self-attention')
        parser.add_argument('--MSA_head_dim', type=int, default=25, help='Head dimension of multihead self-attention')
        parser.add_argument('--attention_dim', type=int, default=256, help="Attention dimension")
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
        parser.add_argument('--graph_depth', type=int, default=3, help='Number of dual-graph modeling layers')
        # SAG config
        parser.add_argument('--SAG_hops', type=int, default=2, help='SAG hops')
        parser.add_argument('--SAG_neighbors', type=int, default=5, help='SAG neighbors')

        self.attribute_dict = dict(vars(parser.parse_args()))
        for attribute in self.attribute_dict:
            setattr(self, attribute, self.attribute_dict[attribute])
        self.seed = self.seed if self.seed >= 0 else (int)(time.time())
        self.train_root = '../%s/train' % self.dataset
        self.dev_root = '../%s/dev' % self.dataset
        self.test_root = '../%s/test' % self.dataset
        if self.dataset == 'MIND-small':
            self.dropout_rate = 0.2
            self.epoch = 16
        if self.dataset == 'MIND-large':
            self.dropout_rate = 0.1
            self.epoch = 7
        self.news_graph_size = 1
        neighbors = 1
        for i in range(self.SAG_hops):
            if i == 0:
                neighbors *= self.SAG_neighbors
            else:
                neighbors *= self.SAG_neighbors - 1
            self.news_graph_size += neighbors
        if self.local_rank in [-1, 0]:
            print('*' * 32 + ' Experiment setting ' + '*' * 32)
            for attribute in self.attribute_dict:
                print(attribute + ' : ' + str(getattr(self, attribute)))
            print('*' * 32 + ' Experiment setting ' + '*' * 32)


    def set_cuda(self):
        assert torch.cuda.is_available(), 'GPU is not available'
        if self.local_rank == -1:
            torch.cuda.set_device(self.device_id)
        else:
            torch.cuda.set_device(torch.device('cuda:{}'.format(self.local_rank)))
            dist.init_process_group(backend='nccl', timeout=datetime.timedelta(0, 43200))
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True # For reproducibility


    def preliminary_setup(self):
        if not self.local_rank in [-1, 0]:
            return
        if self.dataset == 'MIND-small':
            if not os.path.exists('../MIND-small/train') or not os.path.exists('../MIND-small/dev') or not os.path.exists('../MIND-small/test'):
                prepare_MIND_small()
        elif self.dataset == 'MIND-large':
            if not os.path.exists('../MIND-large/train') or not os.path.exists('../MIND-large/dev') or not os.path.exists('../MIND-large/test'):
                prepare_MIND_large()
        else:
            raise Exception('Dataset is chosen from \'MIND-small\' and \'MIND-large\'')
        model_name = self.news_encoder + '-' + self.graph_encoder
        mkdirs = lambda p: os.makedirs(p) if not os.path.exists(p) else None
        mkdirs('configs/' + self.dataset + '/' + model_name)
        mkdirs('models/' + self.dataset + '/' + model_name)
        mkdirs('best_model/' + self.dataset + '/' + model_name)
        mkdirs('dev/' + self.dataset + '/ref')
        mkdirs('dev/' + self.dataset + '/res/' + model_name)
        mkdirs('test/' + self.dataset + '/ref')
        mkdirs('test/' + self.dataset + '/res/' + model_name)
        mkdirs('results/' + self.dataset + '/' + model_name)
        if not os.path.exists('dev/%s/ref/truth.txt' % self.dataset):
            with open(os.path.join(self.dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as dev_f:
                with open('dev/%s/ref/truth.txt' % self.dataset, 'w', encoding='utf-8') as truth_f:
                    for dev_ID, line in enumerate(dev_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                        truth_f.write(('' if dev_ID == 0 else '\n') + str(dev_ID + 1) + ' ' + str(labels).replace(' ', ''))
        # For MIND-small, we perform evaluation on the dataset.
        # For MIND-large, we submit the model prediction to the MIND leadboard website for performance evaluation.
        if self.dataset == 'MIND-small':
            if not os.path.exists('test/MIND-small/ref/truth.txt'):
                with open(os.path.join(self.test_root, 'behaviors.tsv'), 'r', encoding='utf-8') as test_f:
                    with open('test/MIND-small/ref/truth.txt', 'w', encoding='utf-8') as truth_f:
                        for test_ID, line in enumerate(test_f):
                            impression_ID, user_ID, time, history, impressions = line.split('\t')
                            labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                            truth_f.write(('' if test_ID == 0 else '\n') + str(test_ID + 1) + ' ' + str(labels).replace(' ', ''))
        elif self.dataset == 'MIND-large':
            mkdirs('prediction/MIND-large/' + model_name)
        MIND_Corpus.preprocess(self)


    def __init__(self):
        self.parse_argument()
        self.set_cuda()
        self.preliminary_setup()


if __name__ == '__main__':
    config = Config()
