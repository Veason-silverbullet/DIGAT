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
    max_history_num = model.user_encoder.max_history_num
    news_embeddings = torch.zeros([cached_news_num, model.news_encoder.news_embedding_dim]).cuda()
    augmented_news_embeddings = torch.zeros([cached_news_num, model.news_encoder.news_embedding_dim]).cuda()
    news_encoder = model.news_encoder
    user_encoder = model.user_encoder
    index = 0
    # Cache news representations
    with torch.no_grad():
        for (title_text, title_mask, augmented_title_text, augmented_title_mask) in news_dataloader:
            title_text = title_text.cuda(non_blocking=True)
            title_mask = title_mask.cuda(non_blocking=True)
            augmented_title_text = augmented_title_text.cuda(non_blocking=True)
            augmented_title_mask = augmented_title_mask.cuda(non_blocking=True)
            _batch_size = title_text.size(0)
            title_text = title_text.unsqueeze(dim=1)
            title_mask = title_mask.unsqueeze(dim=1)
            augmented_title_text = augmented_title_text.unsqueeze(dim=1)
            augmented_title_mask = augmented_title_mask.unsqueeze(dim=1)
            news_embeddings[index: index+_batch_size] = news_encoder(title_text, title_mask).squeeze(dim=1)
            augmented_news_embeddings[index: index+_batch_size] = news_encoder(title_text, title_mask, augmented_title_text, augmented_title_mask).squeeze(dim=1)
            index += _batch_size
    dataset = MIND_DevTest_Dataset(mind_corpus, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size // 32, pin_memory=True)
    indices = (mind_corpus.dev_indices if mode == 'dev' else mind_corpus.test_indices)
    scores = torch.zeros([len(indices)]).cuda()
    index = 0
    with torch.no_grad():
        for (candidate_news_index, history_news_indices, history_news_mask) in dataloader:
            candidate_news_index = candidate_news_index.cuda(non_blocking=True)
            history_news_indices = history_news_indices.cuda(non_blocking=True)
            history_news_mask = history_news_mask.cuda(non_blocking=True)
            batch_size = candidate_news_index.size(0)
            candidate_news_representations = augmented_news_embeddings.index_select(dim=0, index=candidate_news_index.flatten()).view([batch_size, news_embedding_dim])
            user_news_representaions = news_embeddings.index_select(dim=0, index=history_news_indices.flatten()).view([batch_size, max_history_num, news_embedding_dim])
            user_representations = user_encoder.encode(user_news_representaions, history_news_mask)
            scores[index: index+batch_size] = (candidate_news_representations * user_representations).sum(dim=1, keepdim=False)
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
    torch.cuda.empty_cache()
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
        return '%.4f\nAUC = %.4f\nMRR = %.4f\nnDCG@5 = %.4f\nnDCG@10 = %.4f' % (self.avg, self.auc, self.mrr, self.ndcg5, self.ndcg10)
