from MIND_corpus import MIND_Corpus
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
        self.augmented_news = corpus.augmented_news
        self.train_behaviors = corpus.train_behaviors
        self.train_samples = [[0 for _ in range(1 + self.negative_sample_num)] for __ in range(len(self.train_behaviors))]
        self.num = len(self.train_behaviors)

    def negative_sampling(self, rank=None):
        print('\n%sBegin negative sampling, training sample num : %d' % ('' if rank is None else ('rank ' + str(rank) + ' : '), self.num))
        start_time = time.time()
        for i, train_behavior in enumerate(self.train_behaviors):
            self.train_samples[i][0] = train_behavior[2]
            negative_samples = train_behavior[3]
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
        end_time = time.time()
        print('%sEnd negative sampling, used time : %.3fs' % ('' if rank is None else ('rank ' + str(rank) + ' : '), end_time - start_time))

    # user_title_text           : [max_history_num, max_title_length]
    # user_title_mask           : [max_history_num, max_title_length]
    # user_history_mask         : [max_history_num]
    # news_title_text           : [1 + negative_sample_num, max_title_length]
    # news_title_mask           : [1 + negative_sample_num, max_title_length]
    # augmented_news_title_text : [1 + negative_sample_num, augmented_news_num, max_title_length]
    # augmented_news_title_mask : [1 + negative_sample_num, augmented_news_num, max_title_length]
    def __getitem__(self, index):
        train_behavior = self.train_behaviors[index]
        history_index = train_behavior[0]
        sample_index = self.train_samples[index]
        return self.news_title_text[history_index], self.news_title_mask[history_index], train_behavior[1], \
               self.news_title_text[sample_index], self.news_title_mask[sample_index], self.news_title_text[self.augmented_news[sample_index]], self.news_title_mask[self.augmented_news[sample_index]]

    def __len__(self):
        return self.num


class MIND_DevTest_Dataset(data.Dataset):
    def __init__(self, corpus: MIND_Corpus, mode: str):
        assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
        self.behaviors = corpus.dev_behaviors if mode == 'dev' else corpus.test_behaviors
        self.num = len(self.behaviors)

    # candidate_news_index : [1]
    # history_news_indices : [max_history_num]
    # history_news_mask    : [max_history_num]
    def __getitem__(self, index):
        behavior = self.behaviors[index]
        return behavior[0], behavior[1], behavior[2]

    def __len__(self):
        return self.num


class MIND_News_Dataset(data.Dataset):
    def __init__(self, corpus: MIND_Corpus):
        self.news_title_text = corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.augmented_news = corpus.augmented_news
        self.num = corpus.news_title_text.shape[0]

    # title_text           : [max_title_length]
    # title_mask           : [max_title_length]
    # augmented_title_text : [max_title_length]
    # augmented_title_mask : [max_title_length]
    def __getitem__(self, index):
        augmented_news_index = self.augmented_news[index]
        return self.news_title_text[index], self.news_title_mask[index], self.news_title_text[augmented_news_index], self.news_title_mask[augmented_news_index]

    def __len__(self):
        return self.num


if __name__ == '__main__':
    start_time = time.time()
    config = Config()
    mind_corpus = MIND_Corpus(config)
    print('user_num :', len(mind_corpus.user_ID_dict))
    print('news_num :', len(mind_corpus.news_title_text))
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
    for (user_title_text, user_title_mask, user_history_mask, \
         news_title_text, news_title_mask, augmented_news_title_text, augmented_news_title_mask) in train_dataloader:
        print('user_title_text', user_title_text.size(), user_title_text.dtype)
        print('user_title_mask', user_title_mask.size(), user_title_mask.dtype)
        print('user_history_mask', user_history_mask.size(), user_history_mask.dtype)
        print('news_title_text', news_title_text.size(), news_title_text.dtype)
        print('news_title_mask', news_title_mask.size(), news_title_mask.dtype)
        print('augmented_news_title_text', augmented_news_title_text.size(), augmented_news_title_text.dtype)
        print('augmented_news_title_mask', augmented_news_title_mask.size(), augmented_news_title_mask.dtype)
        break

    print('MIND_Dev_Dataset :', len(mind_dev_dataset))
    dev_dataloader = DataLoader(mind_dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.batch_size // 16)
    for (candidate_news_index, history_news_indices, history_news_mask) in dev_dataloader:
        print('candidate_news_index', candidate_news_index.size(), candidate_news_index.dtype)
        print('history_news_indices', history_news_indices.size(), history_news_indices.dtype)
        print('history_news_mask', history_news_mask.size(), history_news_mask.dtype)
        break
    print(len(mind_corpus.dev_indices))

    print('MIND_Test_Dataset :', len(mind_test_dataset))
    test_dataloader = DataLoader(mind_test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.batch_size // 16)
    for (candidate_news_index, history_news_indices, history_news_mask) in test_dataloader:
        print('candidate_news_index', candidate_news_index.size(), candidate_news_index.dtype)
        print('history_news_indices', history_news_indices.size(), history_news_indices.dtype)
        print('history_news_mask', history_news_mask.size(), history_news_mask.dtype)
        break
    print(len(mind_corpus.test_indices))

    print('MIND_News_Dataset :', len(mind_news_dataset))
    news_dataloader = DataLoader(mind_news_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.batch_size // 16)
    for (news_title_text, news_title_mask, augmented_news_title_text, augmented_news_title_mask) in news_dataloader:
        print('news_title_text', news_title_text.size(), news_title_text.dtype)
        print('news_title_mask', news_title_mask.size(), news_title_mask.dtype)
        print('augmented_news_title_text', augmented_news_title_text.size(), augmented_news_title_text.dtype)
        print('augmented_news_title_mask', augmented_news_title_mask.size(), augmented_news_title_mask.dtype)
        break
