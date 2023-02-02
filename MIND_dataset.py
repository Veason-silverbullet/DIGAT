from MIND_corpus import MIND_Corpus
import numpy as np
import time
from config import Config
import torch.utils.data as data
from numpy.random import randint
from torch.utils.data import DataLoader


class MIND_Train_Dataset(data.Dataset):
    def __init__(self, corpus: MIND_Corpus):
        self.negative_sample_num = corpus.negative_sample_num
        self.news_title_text =  corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_node_ID = corpus.news_node_ID
        self.news_graph = corpus.news_graph
        self.news_graph_mask = corpus.news_graph_mask
        self.user_history_graph = corpus.train_user_history_graph
        self.user_history_graph_mask = corpus.train_user_history_graph_mask
        self.user_history_category_mask = corpus.train_user_history_category_mask
        self.user_history_category_indices = corpus.train_user_history_category_indices
        self.train_behaviors = corpus.train_behaviors
        self.train_samples = [[0 for _ in range(1 + self.negative_sample_num)] for __ in range(len(self.train_behaviors))]
        self.num = len(self.train_behaviors)

    def negative_sampling(self, verbose=True):
        if verbose:
            print('\nBegin negative sampling, training sample num : %d' % self.num)
            start_time = time.time()
        for i, train_behavior in enumerate(self.train_behaviors):
            self.train_samples[i][0] = train_behavior[1]
            negative_samples = train_behavior[2]
            news_num = len(negative_samples)
            if news_num <= self.negative_sample_num:
                for j in range(self.negative_sample_num):
                    self.train_samples[i][j + 1] = negative_samples[j % news_num]
            else:
                used_negative_samples = set()
                for j in range(self.negative_sample_num):
                    while True:
                        k = randint(0, news_num)
                        if k not in used_negative_samples:
                            self.train_samples[i][j + 1] = negative_samples[k]
                            used_negative_samples.add(k)
                            break
        if verbose:
            end_time = time.time()
            print('End negative sampling, used time : %.3fs' % (end_time - start_time))

    # user_title_text       : [max_history_num, max_title_length]
    # user_title_mask       : [max_history_num, max_title_length]
    # user_graph            : [user_graph_size, user_graph_size]
    # user_graph_mask       : [user_graph_size]
    # user_category_mask    : [category_num + 1]
    # user_category_indices : [max_history_num]
    # news_title_text       : [1 + negative_sample_num, news_graph_size, max_title_length]
    # news_title_mask       : [1 + negative_sample_num, news_graph_size, max_title_length]
    # news_graph            : [1 + negative_sample_num, news_graph_size, news_graph_size]
    # news_graph_mask       : [1 + negative_sample_num, news_graph_size]
    def __getitem__(self, index):
        train_behavior = self.train_behaviors[index]
        history_index = train_behavior[0]
        sample_index = self.train_samples[index]
        news_graph_index = self.news_node_ID[sample_index]
        behavior_index = train_behavior[3]
        return self.news_title_text[history_index], self.news_title_mask[history_index], self.user_history_graph[behavior_index], self.user_history_graph_mask[behavior_index], self.user_history_category_mask[behavior_index], self.user_history_category_indices[behavior_index], \
               self.news_title_text[news_graph_index], self.news_title_mask[news_graph_index], self.news_graph[sample_index], self.news_graph_mask[sample_index]

    def __len__(self):
        return self.num


class MIND_DevTest_Dataset(data.Dataset):
    def __init__(self, corpus: MIND_Corpus, mode: str):
        assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
        self.news_title_text =  corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_graph = corpus.news_graph
        self.news_graph_mask = corpus.news_graph_mask
        self.user_history_graph = corpus.dev_user_history_graph if mode == 'dev' else corpus.test_user_history_graph
        self.user_history_graph_mask = corpus.dev_user_history_graph_mask if mode == 'dev' else corpus.test_user_history_graph_mask
        self.user_history_category_mask = corpus.dev_user_history_category_mask if mode == 'dev' else corpus.test_user_history_category_mask
        self.user_history_category_indices = corpus.dev_user_history_category_indices if mode == 'dev' else corpus.test_user_history_category_indices
        self.behaviors = corpus.dev_behaviors if mode == 'dev' else corpus.test_behaviors
        news_node_ID = corpus.news_node_ID
        self.candidate_news_ID = [self.behaviors[i][1] for i in range(len(self.behaviors))]
        self.num = len(self.behaviors)

    # user_title_index      : [max_history_num]
    # user_graph            : [user_graph_size, user_graph_size]
    # user_graph_mask       : [user_graph_size]
    # user_category_mask    : [category_num + 1]
    # user_category_indices : [max_history_num]
    # candidate_news_ID     : [news_graph_size]
    # news_graph            : [news_graph_size, news_graph_size]
    # news_graph_mask       : [news_graph_size]
    def __getitem__(self, index):
        behavior = self.behaviors[index]
        candidate_news_index = behavior[1]
        behavior_index = behavior[2]
        return np.array(behavior[0], dtype=np.int32), self.user_history_graph[behavior_index], self.user_history_graph_mask[behavior_index], self.user_history_category_mask[behavior_index], self.user_history_category_indices[behavior_index], \
               self.candidate_news_ID[index], self.news_graph[candidate_news_index], self.news_graph_mask[candidate_news_index]

    def __len__(self):
        return self.num


class MIND_News_Dataset(data.Dataset):
    def __init__(self, corpus: MIND_Corpus):
        self.news_title_text = corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.num = self.news_title_text.shape[0]

    # title_text : [max_title_length]
    # title_mask : [max_title_length]
    def __getitem__(self, index):
        return self.news_title_text[index], self.news_title_mask[index]

    def __len__(self):
        return self.num


if __name__ == '__main__':
    start_time = time.time()
    config = Config()
    mind_corpus = MIND_Corpus(config)
    print('user_num :', len(mind_corpus.user_ID_dict))
    print('news_num :', len(mind_corpus.news_ID_dict))
    print('average title word num :', mind_corpus.title_word_num / mind_corpus.news_num)
    mind_train_dataset = MIND_Train_Dataset(mind_corpus)
    mind_dev_dataset = MIND_DevTest_Dataset(mind_corpus, 'dev')
    mind_test_dataset = MIND_DevTest_Dataset(mind_corpus, 'test')
    mind_news_dataset = MIND_News_Dataset(mind_corpus)
    mind_train_dataset.negative_sampling()
    end_time = time.time()
    print('load time : %.3fs' % (end_time - start_time))
    print('MIND_Train_Dataset :', len(mind_train_dataset))
    train_dataloader = DataLoader(mind_train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.batch_size // 16)
    for (user_title_text, user_title_mask, user_graph, user_graph_mask, user_category_mask, user_category_indices, \
         news_title_text, news_title_mask, news_graph, news_graph_mask) in train_dataloader:
        print('user_title_text', user_title_text.size(), user_title_text.dtype)
        print('user_title_mask', user_title_mask.size(), user_title_mask.dtype)
        print('user_graph', user_graph.size(), user_graph.dtype)
        print('user_graph_mask', user_graph_mask.size(), user_graph_mask.dtype)
        print('user_category_mask', user_category_mask.size(), user_category_mask.dtype)
        print('user_category_indices', user_category_indices.size(), user_category_indices.dtype)
        print('news_title_text', news_title_text.size(), news_title_text.dtype)
        print('news_title_mask', news_title_mask.size(), news_title_mask.dtype)
        print('news_graph', news_graph.size(), news_graph.dtype)
        print('news_graph_mask', news_graph_mask.size(), news_graph_mask.dtype)
        break

    print('MIND_Dev_Dataset :', len(mind_dev_dataset))
    dev_dataloader = DataLoader(mind_dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.batch_size // 16)
    for (user_title_index, user_graph, user_graph_mask, user_category_mask, user_category_indices, \
         news_ID, news_graph, news_graph_mask) in dev_dataloader:
        print('user_title_index', user_title_index.size(), user_title_index.dtype)
        print('user_graph', user_graph.size(), user_graph.dtype)
        print('user_graph_mask', user_graph_mask.size(), user_graph_mask.dtype)
        print('user_category_mask', user_category_mask.size(), user_category_mask.dtype)
        print('user_category_indices', user_category_indices.size(), user_category_indices.dtype)
        print('news_ID', news_ID.size(), news_ID.dtype)
        print('news_graph', news_graph.size(), news_graph.dtype)
        print('news_graph_mask', news_graph_mask.size(), news_graph_mask.dtype)
        break
    print(len(mind_corpus.dev_indices))

    print('MIND_Test_Dataset :', len(mind_test_dataset))
    test_dataloader = DataLoader(mind_test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.batch_size // 16)
    for (user_title_index, user_graph, user_graph_mask, user_category_mask, user_category_indices, \
         news_ID, news_graph, news_graph_mask) in test_dataloader:
        print('user_title_index', user_title_index.size(), user_title_index.dtype)
        print('user_graph', user_graph.size(), user_graph.dtype)
        print('user_graph_mask', user_graph_mask.size(), user_graph_mask.dtype)
        print('user_category_mask', user_category_mask.size(), user_category_mask.dtype)
        print('user_category_indices', user_category_indices.size(), user_category_indices.dtype)
        print('news_ID', news_ID.size(), news_ID.dtype)
        print('news_graph', news_graph.size(), news_graph.dtype)
        print('news_graph_mask', news_graph_mask.size(), news_graph_mask.dtype)
        break
    print(len(mind_corpus.test_indices))

    print('MIND_News_Dataset :', len(mind_news_dataset))
    news_dataloader = DataLoader(mind_news_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.batch_size // 16)
    for (news_title_text, news_title_mask) in news_dataloader:
        print('news_title_text', news_title_text.size(), news_title_text.dtype)
        print('news_title_mask', news_title_mask.size(), news_title_mask.dtype)
        break
