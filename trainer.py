import os
import shutil
import json
from config import Config
from MIND_corpus import MIND_Corpus
from MIND_dataset import MIND_Train_Dataset
from util import AvgMetric, get_run_index, compute_scores
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


class Trainer:
    def __init__(self, model: nn.Module, config: Config, mind_corpus: MIND_Corpus):
        self.model = model if config.local_rank == -1 else DDP(model, device_ids=[config.local_rank], output_device=config.local_rank)
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.max_history_num = config.max_history_num
        self.negative_sample_num = config.negative_sample_num
        self.lr = config.lr
        no_decay = ['.bias', 'embed', 'graph_encoder.']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n.lower() for nd in no_decay) and p.requires_grad], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n.lower() for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        self.optimizer = optim.Adam(optimizer_grouped_parameters, lr=config.lr)
        self.gradient_clip_norm = config.gradient_clip_norm
        self.lr_decay_epoch = (self.epoch - 1) // 10 + 1 # 10% learning rate decay
        self.mind_corpus = mind_corpus
        self.train_dataset = MIND_Train_Dataset(mind_corpus)
        self.local_rank = config.local_rank
        self.is_main_rank = self.local_rank in [-1, 0]
        if self.is_main_rank:
            self.dataset_type = config.dataset
            self.run_index = get_run_index(config.dataset, model.model_name)
            config.run_index = self.run_index
            if not os.path.exists('models/' + config.dataset + '/' + model.model_name + '/#' + str(self.run_index)):
                os.mkdir('models/' + config.dataset + '/' + model.model_name + '/#' + str(self.run_index))
            if not os.path.exists('best_model/' + config.dataset + '/' + model.model_name + '/#' + str(self.run_index)):
                os.mkdir('best_model/' + config.dataset + '/' + model.model_name + '/#' + str(self.run_index))
            if not os.path.exists('dev/' + config.dataset + '/res/' + model.model_name + '/#' + str(self.run_index)):
                os.mkdir('dev/' + config.dataset + '/res/' + model.model_name + '/#' + str(self.run_index))
            with open('configs/' + config.dataset + '/' + model.model_name + '/#' + str(self.run_index) + '.json', 'w', encoding='utf-8') as f:
                json.dump(config.attribute_dict, f)
                self.attribute_dict = config.attribute_dict
            if config.dataset == 'MIND-large' and not os.path.exists('prediction/' + config.dataset + '/' + model.model_name + '/#' + str(self.run_index)):
                os.mkdir('prediction/' + config.dataset + '/' + model.model_name + '/#' + str(self.run_index))
            self.dev_criterion = config.dev_criterion
            self.early_stopping_epoch = config.early_stopping_epoch
            self.auc = []
            self.mrr = []
            self.ndcg5 = []
            self.ndcg10 = []
            self.best_dev_epoch = 0
            self.best_dev_auc = 0
            self.best_dev_mrr = 0
            self.best_dev_ndcg5 = 0
            self.best_dev_ndcg10 = 0
            self.best_dev_avg = AvgMetric(0, 0, 0, 0)
            self.epoch_not_increase = 0
            print('Running : ' + (self.model.module.model_name if hasattr(self.model, 'module') else self.model.model_name) + '\t#' + str(self.run_index))

    def lr_decay(self):
        for group in self.optimizer.param_groups:
            group['lr'] = group['lr'] / 10

    def train(self):
        model = self.model
        for e in (tqdm(range(1, self.epoch + 1)) if self.is_main_rank else range(1, self.epoch + 1)):
            self.train_dataset.negative_sampling(verbose=self.is_main_rank)
            if self.local_rank == -1:
                train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.batch_size // 16, pin_memory=True)
            else:
                train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
                train_sampler.set_epoch(e)
                train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.batch_size // 16, pin_memory=True, sampler=train_sampler)
            if self.lr_decay_epoch > 0 and e == self.epoch - self.lr_decay_epoch + 1:
                self.lr_decay()
            model.train()
            if self.is_main_rank:
                epoch_loss = 0
            for (user_title_text, user_title_mask, user_graph, user_graph_mask, user_category_mask, user_category_indices, \
                 news_title_text, news_title_mask, news_graph, news_graph_mask) in train_dataloader:
                user_title_text = user_title_text.cuda(non_blocking=True)                     # [batch_size, max_history_num, max_title_length]
                user_title_mask = user_title_mask.cuda(non_blocking=True)                     # [batch_size, max_history_num, max_title_length]
                user_graph = user_graph.cuda(non_blocking=True)                               # [batch_size, max_history_num, max_history_num]
                user_category_mask = user_category_mask.cuda(non_blocking=True)               # [batch_size, category_num]
                user_category_indices = user_category_indices.cuda(non_blocking=True)         # [batch_size, max_history_num]
                news_title_text = news_title_text.cuda(non_blocking=True)                     # [batch_size, 1 + negative_sample_num, max_title_length]
                news_title_mask = news_title_mask.cuda(non_blocking=True)                     # [batch_size, 1 + negative_sample_num, max_title_length]
                news_graph = news_graph.cuda(non_blocking=True)                               # [batch_size, 1 + negative_sample_num, news_graph_size, news_graph_size]
                news_graph_mask = news_graph_mask.cuda(non_blocking=True)                     # [batch_size, 1 + negative_sample_num, news_graph_size]

                logits = model(user_title_text, user_title_mask, user_graph, user_category_mask, user_category_indices, \
                               news_title_text, news_title_mask, news_graph, news_graph_mask) # [batch_size, 1 + negative_sample_num]
                loss = (-torch.log_softmax(logits, dim=1).select(1, 0)).mean()
                self.optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
                if self.is_main_rank:
                    epoch_loss += loss.item()

            if self.is_main_rank:
                print('Epoch %d : train done' % e)
                print('loss =', epoch_loss / len(train_dataloader))
                # dev
                val_model = model.module if hasattr(model, 'module') else model
                auc, mrr, ndcg5, ndcg10 = compute_scores(val_model, self.mind_corpus, self.batch_size * 16, self.dataset_type, 'dev', 'dev/' + self.dataset_type + '/res/' + val_model.model_name + '/#' + str(self.run_index) + '/' + val_model.model_name + '-' + str(e) + '.txt')
                self.auc.append(auc)
                self.mrr.append(mrr)
                self.ndcg5.append(ndcg5)
                self.ndcg10.append(ndcg10)
                print('Epoch %d : dev done\nDev criterions' % e)
                print('AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5  = {:.4f}\nnDCG@10 = {:.4f}'.format(auc, mrr, ndcg5, ndcg10))
                if self.dev_criterion == 'auc':
                    if auc >= self.best_dev_auc:
                        self.best_dev_auc = auc
                        self.best_dev_epoch = e
                        with open('results/' + self.dataset_type + '/' + val_model.model_name + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                            result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                        self.epoch_not_increase = 0
                    else:
                        self.epoch_not_increase += 1
                elif self.dev_criterion == 'mrr':
                    if mrr >= self.best_dev_mrr:
                        self.best_dev_mrr = mrr
                        self.best_dev_epoch = e
                        with open('results/' + self.dataset_type + '/' + val_model.model_name + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                            result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                        self.epoch_not_increase = 0
                    else:
                        self.epoch_not_increase += 1
                elif self.dev_criterion == 'ndcg5':
                    if ndcg5 >= self.best_dev_ndcg5:
                        self.best_dev_ndcg5 = ndcg5
                        self.best_dev_epoch = e
                        with open('results/' + self.dataset_type + '/' + val_model.model_name + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                            result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                        self.epoch_not_increase = 0
                    else:
                        self.epoch_not_increase += 1
                elif self.dev_criterion == 'ndcg10':
                    if ndcg10 >= self.best_dev_ndcg10:
                        self.best_dev_ndcg10 = ndcg10
                        self.best_dev_epoch = e
                        with open('results/' + self.dataset_type + '/' + val_model.model_name + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                            result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                        self.epoch_not_increase = 0
                    else:
                        self.epoch_not_increase += 1
                else:
                    avg = AvgMetric(auc, mrr, ndcg5, ndcg10)
                    if avg >= self.best_dev_avg:
                        self.best_dev_avg = avg
                        self.best_dev_epoch = e
                        with open('results/' + self.dataset_type + '/' + val_model.model_name + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                            result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                        self.epoch_not_increase = 0
                    else:
                        self.epoch_not_increase += 1
                print('Best epoch :', self.best_dev_epoch)
                print('Best ' + self.dev_criterion + ' : ' + str(getattr(self, 'best_dev_' + self.dev_criterion)))
                if self.epoch_not_increase == 0:
                    torch.save({val_model.model_name: val_model.state_dict()}, 'models/' + self.dataset_type + '/' + val_model.model_name + '/#' + str(self.run_index) + '/' + val_model.model_name + '-' + str(e))
                if self.epoch_not_increase > self.early_stopping_epoch:
                    break
            torch.cuda.empty_cache()

        if self.is_main_rank:
            if hasattr(self.model, 'module'):
                self.model = self.model.module
            with open('dev/%s/res/%s/#%d/%s-dev_log.txt' % (self.dataset_type, self.model.model_name, self.run_index, self.model.model_name), 'w', encoding='utf-8') as f:
                f.write('Epoch\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
                for i in range(len(self.auc)):
                    f.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\n' % (i + 1, self.auc[i], self.mrr[i], self.ndcg5[i], self.ndcg10[i]))
                f.write('Best dev epoch : ' + str(self.best_dev_epoch))
            print('Training : ' + self.model.model_name + ' #' + str(self.run_index) + ' completed\nDev criterions:')
            print('AUC : %.4f' % self.auc[self.best_dev_epoch - 1])
            print('MRR : %.4f' % self.mrr[self.best_dev_epoch - 1])
            print('nDCG@5  : %.4f' % self.ndcg5[self.best_dev_epoch - 1])
            print('nDCG@10 : %.4f' % self.ndcg10[self.best_dev_epoch - 1])
            shutil.copy('models/' + self.dataset_type + '/' + self.model.model_name + '/#' + str(self.run_index) + '/' + self.model.model_name + '-' + str(self.best_dev_epoch), 'best_model/' + self.dataset_type + '/' + self.model.model_name + '/#' + str(self.run_index) + '/' + self.model.model_name)
