import os
import torch
import torch.nn as nn
from MIND_corpus import MIND_Corpus
from MIND_dataset import MIND_News_Dataset, MIND_DevTest_Dataset
from torch.utils.data import DataLoader
from evaluate import scoring


def compute_scores(model: nn.Module, mind_corpus: MIND_Corpus, batch_size: int, dataset_type: str, mode: str, result_file: str):
    assert dataset_type in ['MIND-small', 'MIND-large'], 'dataset_type must be chosen from \'MIND-small\' or \'MIND-large\''
    assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
    torch.cuda.empty_cache()
    model.eval()
    news_dataset = MIND_News_Dataset(mind_corpus)
    news_dataloader = DataLoader(news_dataset, batch_size=batch_size * 4, shuffle=False, num_workers=0, pin_memory=True)
    cached_news_num = len(news_dataset)
    news_embedding_dim = model.news_encoder.news_embedding_dim
    max_history_num = model.graph_encoder.max_history_num
    cached_news_representations = torch.zeros([cached_news_num, news_embedding_dim]).cuda() # [cached_news_num, news_embedding_dim]
    news_encoder = model.news_encoder
    graph_encoder = model.graph_encoder
    # Cache news representations
    with torch.no_grad():
        index = 0
        for (title_text, title_mask) in news_dataloader:
            title_text = title_text.cuda(non_blocking=True)
            title_mask = title_mask.cuda(non_blocking=True)
            _batch_size = title_text.size(0)
            title_text = title_text.unsqueeze(dim=1)
            title_mask = title_mask.unsqueeze(dim=1)
            cached_news_representations[index: index+_batch_size] = news_encoder(title_text, title_mask).squeeze(dim=1)
            index += _batch_size
        news_node_index = torch.from_numpy(mind_corpus.news_node_ID).cuda()                                                                                               # [cached_news_num, SA_news_num]
        news_graph_masks = torch.from_numpy(mind_corpus.news_graph_mask).cuda()                                                                                           # [cached_news_num, SA_news_num, SA_news_num]
        cached_SA_news_representations = cached_news_representations.index_select(dim=0, index=news_node_index.flatten()).view([cached_news_num, -1, news_embedding_dim]) # [cached_news_num, SA_news_num, news_embedding_dim]
        cached_c_n0 = torch.zeros([cached_news_num, news_embedding_dim]).cuda()                                                                                           # [cached_news_num, news_embedding_dim]
        index = 0
        if model.model_name.split('-')[1] in ['DIGAT', 'wo_interaction', 'news_graph_wo_inter', 'user_graph_wo_inter']:
            while index != cached_news_num:
                _index = min(index + batch_size, cached_news_num)
                batch_num = _index - index
                cached_c_n0[index:_index] = graph_encoder.compute_news_graph_context(torch.narrow(cached_SA_news_representations, 0, index, batch_num), torch.narrow(news_graph_masks, 0, index, batch_num))
                index = _index
        elif model.model_name.split('-')[1] == 'Seq_SA':
            while index != cached_news_num:
                _index = min(index + batch_size, cached_news_num)
                batch_num = _index - index
                cached_c_n0[index:_index] = graph_encoder.compute_news_sequence_context(torch.narrow(cached_SA_news_representations, 0, index, batch_num), torch.narrow(news_graph_masks, 0, index, batch_num))
                index = _index
        dataset = MIND_DevTest_Dataset(mind_corpus, mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size // 32, pin_memory=True)
        indices = (mind_corpus.dev_indices if mode == 'dev' else mind_corpus.test_indices)
        scores = torch.zeros([len(indices)]).cuda()
        index = 0
        for (user_title_index, user_graph, user_graph_mask, user_category_mask, user_category_indices, news_ID, news_graph, news_graph_mask) in dataloader:
            user_title_index = user_title_index.cuda(non_blocking=True)
            user_graph = user_graph.cuda(non_blocking=True)
            user_category_mask = user_category_mask.cuda(non_blocking=True)
            user_category_indices = user_category_indices.cuda(non_blocking=True)
            news_ID = news_ID.cuda(non_blocking=True)
            news_graph = news_graph.cuda(non_blocking=True)
            news_graph_mask = news_graph_mask.cuda(non_blocking=True)
            batch_size = user_title_index.size(0)
            user_representations = cached_news_representations.index_select(dim=0, index=user_title_index.flatten()).view([batch_size, max_history_num, news_embedding_dim])
            news_representations = cached_SA_news_representations.index_select(dim=0, index=news_ID)
            c_n0 = cached_c_n0.index_select(dim=0, index=news_ID)
            scores[index: index+batch_size] = model.inference(user_representations, user_graph, user_category_mask, user_category_indices, news_representations, news_graph, news_graph_mask, c_n0) # [batch_size]
            index += batch_size
    scores = scores.tolist()
    sub_scores = [[] for _ in range(indices[-1] + 1)]
    for i, index in enumerate(indices):
        sub_scores[index].append([scores[i], len(sub_scores[index])])
    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0 for _ in range(len(sub_score))]
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
    if dataset_type == 'MIND-large' and mode == 'test': # Instead of offline evaluation, we submit the MIND-large test result to leaderboard for evaluation
        return None, None, None, None
    with open(mode + '/' + dataset_type + '/ref/truth.txt', 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
        auc, mrr, ndcg, ndcg10 = scoring(truth_f, result_f)
    return auc, mrr, ndcg, ndcg10


def get_run_index(dataset: str, model_name: str):
    assert os.path.exists('results/' + dataset + '/' + model_name), 'result directory does not exist'
    max_index = 0
    for result_file in os.listdir('results/' + dataset + '/' + model_name):
        if result_file.strip()[0] == '#' and result_file.strip()[-4:] == '-dev':
            index = int(result_file.strip()[1:-4])
            max_index = max(index, max_index)
    with open('results/' + dataset + '/' + model_name + '/#' + str(max_index + 1) + '-dev', 'w', encoding='utf-8') as result_f:
        pass
    return max_index + 1


class AvgMetric:
    def __init__(self, auc, mrr, ndcg5, ndcg10):
        self.auc = auc
        self.mrr = mrr
        self.ndcg5 = ndcg5
        self.ndcg10 = ndcg10
        self.avg = (self.auc + self.mrr + (self.ndcg5 + self.ndcg10) / 2) / 3

    def __gt__(self, value):
        return self.avg > value.avg

    def __ge__(self, value):
        return self.avg >= value.avg

    def __lt__(self, value):
        return self.avg < value.avg

    def __le__(self, value):
        return self.avg <= value.avg

    def __str__(self):
        return '%.4f\nAUC = %.4f\nMRR = %.4f\nnDCG@5  = %.4f\nnDCG@10 = %.4f' % (self.avg, self.auc, self.mrr, self.ndcg5, self.ndcg10)
