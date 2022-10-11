import os
import torch
import numpy as np
import json
import pickle
import collections
import re
from torchtext.vocab import GloVe
from config import Config
from build_SA_news_sequence import construct_SA_news_sequence


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

pat = re.compile(r"[\w]+|[.,!?;|]")


class MIND_Corpus:
    @staticmethod
    def preprocess(config: Config):
        user_ID_file = 'user_ID-%s.json' % config.dataset
        news_ID_file = 'news_ID-%s.json' % config.dataset
        category_file = 'category-%s.json' % config.dataset
        subCategory_file = 'subCategory-%s.json' % config.dataset
        vocabulary_file = 'vocabulary-' + str(config.word_threshold) + '-' + str(config.max_title_length) + '-' + config.dataset + '.json'
        word_embedding_file = 'word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + str(config.max_title_length) + '-' + config.dataset + '.pkl'
        augmented_news_file = 'semantic_augmented_news-%d-%s.pkl' % (config.augmented_news_num, config.dataset)
        preprocessed_data_files = [user_ID_file, news_ID_file, category_file, subCategory_file, vocabulary_file, word_embedding_file, augmented_news_file]

        if not all(list(map(os.path.exists, preprocessed_data_files))):
            user_ID_dict = {'<UNK>': 0}
            news_ID_dict = {'<PAD>': 0}
            category_dict = {}
            subCategory_dict = {}
            word_dict = {'<PAD>': 0, '<UNK>': 1}
            word_counter = collections.Counter()

            # 1. user ID dictionay
            with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as train_behaviors_f:
                for line in train_behaviors_f:
                    impression_ID, user_ID, time, history, impressions = line.split('\t')
                    if user_ID not in user_ID_dict:
                        user_ID_dict[user_ID] = len(user_ID_dict)
                with open(user_ID_file, 'w', encoding='utf-8') as user_ID_f:
                    json.dump(user_ID_dict, user_ID_f)

            # 2. news ID dictionay & news category dictionay & news subCategory dictionay
            for i, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                with open(os.path.join(prefix, 'news.tsv'), 'r', encoding='utf-8') as news_f:
                    for line in news_f:
                        news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                        if news_ID not in news_ID_dict:
                            news_ID_dict[news_ID] = len(news_ID_dict)
                            if category not in category_dict:
                                category_dict[category] = len(category_dict)
                            if subCategory not in subCategory_dict:
                                subCategory_dict[subCategory] = len(subCategory_dict)
                            words = pat.findall(title.lower())
                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0: # training set
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter: # already appeared in training set
                                            word_counter[word] += 1
            with open(news_ID_file, 'w', encoding='utf-8') as news_ID_f:
                json.dump(news_ID_dict, news_ID_f)
            with open(category_file, 'w', encoding='utf-8') as category_f:
                json.dump(category_dict, category_f)
            with open(subCategory_file, 'w', encoding='utf-8') as subCategory_f:
                json.dump(subCategory_dict, subCategory_f)

            # 3. word dictionay
            word_counter_list = [[word, word_counter[word]] for word in word_counter]
            word_counter_list.sort(key=lambda x: x[1], reverse=True) # sort by word frequency
            filtered_word_counter_list = list(filter(lambda x: x[1] >= config.word_threshold, word_counter_list))
            for i, word in enumerate(filtered_word_counter_list):
                word_dict[word[0]] = i + 2
            with open(vocabulary_file, 'w', encoding='utf-8') as vocabulary_f:
                json.dump(word_dict, vocabulary_f)

            # 4. Glove word embedding
            if config.word_embedding_dim == 300:
                glove = GloVe(name='840B', dim=300, cache='../../glove', max_vectors=10000000000)
            else:
                glove = GloVe(name='6B', dim=config.word_embedding_dim, cache='../../glove', max_vectors=10000000000)
            glove_stoi = glove.stoi
            glove_vectors = glove.vectors
            glove_mean = torch.mean(glove_vectors, dim=0, keepdim=False)
            glove_std = torch.std(glove_vectors, dim=0, keepdim=False, unbiased=True)
            word_embedding_vectors = torch.zeros([len(word_dict), config.word_embedding_dim])
            word_embedding_vectors[0] = glove_mean
            for word in word_dict:
                index = word_dict[word]
                if index != 0:
                    if word in glove_stoi:
                        word_embedding_vectors[index] = glove_vectors[glove_stoi[word]]
                    else:
                        word_embedding_vectors[index] = torch.normal(mean=glove_mean, std=glove_std)
            with open(word_embedding_file, 'wb') as word_embedding_f:
                pickle.dump(word_embedding_vectors, word_embedding_f)

            # 5. Semantic-augmented news
            if not os.path.exists(augmented_news_file):
                augmented_news = construct_SA_news_sequence(config.dataset, config.train_root, config.dev_root, config.test_root, config.augmented_news_num, news_ID_dict)
                semantic_augmented_news = np.zeros(shape=[len(news_ID_dict), config.augmented_news_num], dtype=np.int32)
                for news_ID in news_ID_dict:
                    index = news_ID_dict[news_ID]
                    if index != 0:
                        augmented_news_list = augmented_news[news_ID]
                        for j in range(min(config.augmented_news_num, len(augmented_news_list))):
                            semantic_augmented_news[index][j] = news_ID_dict[augmented_news_list[j][0]]
                with open(augmented_news_file, 'wb') as f:
                    pickle.dump(semantic_augmented_news, f)


    def __init__(self, config: Config):
        # preprocess data
        MIND_Corpus.preprocess(config)
        with open('user_ID-' + str(config.dataset) + '.json', 'r', encoding='utf-8') as user_ID_f:
            self.user_ID_dict = json.load(user_ID_f)
            config.user_num = len(self.user_ID_dict)
        with open('news_ID-' + str(config.dataset) + '.json', 'r', encoding='utf-8') as news_ID_f:
            self.news_ID_dict = json.load(news_ID_f)
            self.news_num = len(self.news_ID_dict)
        with open('category-' + str(config.dataset) + '.json', 'r', encoding='utf-8') as category_f:
            self.category_dict = json.load(category_f)
            config.category_num = len(self.category_dict)
        with open('subCategory-' + str(config.dataset) + '.json', 'r', encoding='utf-8') as subCategory_f:
            self.subCategory_dict = json.load(subCategory_f)
            config.subCategory_num = len(self.subCategory_dict)
        with open('vocabulary-' + str(config.word_threshold) + '-' + str(config.max_title_length) + '-' + str(config.dataset) + '.json', 'r', encoding='utf-8') as vocabulary_f:
            self.word_dict = json.load(vocabulary_f)
            config.vocabulary_size = len(self.word_dict)
        with open('semantic_augmented_news-%d-%s.pkl' % (config.augmented_news_num, config.dataset), 'rb') as augmented_news_f:
            self.augmented_news = pickle.load(augmented_news_f)

        # meta data
        self.dataset_type = config.dataset
        assert self.dataset_type in ['MIND-small', 'MIND-large'], 'Dataset is chosen from \'MIND-small\' and \'MIND-large\''
        self.negative_sample_num = config.negative_sample_num                                   # negative sample number for training
        self.max_history_num = config.max_history_num                                           # max history number for each training user
        self.max_title_length = config.max_title_length                                         # max title length for each news text
        self.news_title_text = np.zeros([self.news_num, self.max_title_length], dtype=np.int32) # [news_num, max_title_length]
        self.news_title_mask = np.zeros([self.news_num, self.max_title_length], dtype=bool)     # [news_num, max_title_length]
        self.train_behaviors = []                                                               # [[history], [history_mask], click impression, [non-click impressions], behavior_index]
        self.dev_behaviors = []                                                                 # [candidate_news_ID, [history], [history_mask]]
        self.dev_indices = []                                                                   # index for dev
        self.test_behaviors = []                                                                # [candidate_news_ID, [history], [history_mask]]
        self.test_indices = []                                                                  # index for test
        self.title_word_num = 0

        # generate news meta data
        news_ID_set = set(['<PAD>'])
        news_lines = []
        with open(os.path.join(config.train_root, 'news.tsv'), 'r', encoding='utf-8') as train_news_f:
            for line in train_news_f:
                news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                if news_ID not in news_ID_set:
                    news_lines.append(line)
                    news_ID_set.add(news_ID)
        with open(os.path.join(config.dev_root, 'news.tsv'), 'r', encoding='utf-8') as dev_news_f:
            for line in dev_news_f:
                news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                if news_ID not in news_ID_set:
                    news_lines.append(line)
                    news_ID_set.add(news_ID)
        with open(os.path.join(config.test_root, 'news.tsv'), 'r', encoding='utf-8') as test_news_f:
            for line in test_news_f:
                news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                if news_ID not in news_ID_set:
                    news_lines.append(line)
                    news_ID_set.add(news_ID)
        assert self.news_num == len(news_ID_set), 'news num mismatch %d v.s. %d' % (self.news_num, len(news_ID_set))
        for line in news_lines:
            news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
            index = self.news_ID_dict[news_ID]
            words = pat.findall(title.lower())
            for i, word in enumerate(words):
                if i == self.max_title_length:
                    break
                if is_number(word):
                    self.news_title_text[index][i] = self.word_dict['<NUM>']
                elif word in self.word_dict:
                    self.news_title_text[index][i] = self.word_dict[word]
                else:
                    self.news_title_text[index][i] = self.word_dict['<UNK>']
                self.news_title_mask[index][i] = 1
            self.title_word_num += len(words)
        self.news_title_mask[0][0] = 1 # for <PAD> news

        # generate behavior meta data
        with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as train_behaviors_f:
            for behavior_index, line in enumerate(train_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                click_impressions = []
                non_click_impressions = []
                for impression in impressions.strip().split(' '):
                    if impression[-2:] == '-1':
                        click_impressions.append(self.news_ID_dict[impression[:-2]])
                    else:
                        non_click_impressions.append(self.news_ID_dict[impression[:-2]])
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = history[-self.max_history_num:] + [0] * padding_num
                    user_history_mask = np.zeros([self.max_history_num], dtype=bool)
                    user_history_mask[:min(len(history), self.max_history_num)] = True
                    for click_impression in click_impressions:
                        self.train_behaviors.append([user_history, user_history_mask, click_impression, non_click_impressions, behavior_index])
                else:
                    for click_impression in click_impressions:
                        self.train_behaviors.append([[0 for _ in range(self.max_history_num)], np.zeros([self.max_history_num], dtype=bool), click_impression, non_click_impressions, behavior_index])
        with open(os.path.join(config.dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as dev_behaviors_f:
            for dev_ID, line in enumerate(dev_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = np.array(history[-self.max_history_num:] + [0] * padding_num, dtype=np.int32)
                    user_history_mask = np.zeros([self.max_history_num], dtype=bool)
                    user_history_mask[:min(len(history), self.max_history_num)] = True
                    for impression in impressions.strip().split(' '):
                        self.dev_indices.append(dev_ID)
                        self.dev_behaviors.append([self.news_ID_dict[impression[:-2]], user_history, user_history_mask])
                else:
                    for impression in impressions.strip().split(' '):
                        self.dev_indices.append(dev_ID)
                        self.dev_behaviors.append([self.news_ID_dict[impression[:-2]], np.zeros([self.max_history_num], dtype=np.int32), np.zeros([self.max_history_num], dtype=bool)])
        with open(os.path.join(config.test_root, 'behaviors.tsv'), 'r', encoding='utf-8') as test_behaviors_f:
            for test_ID, line in enumerate(test_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = np.array(history[-self.max_history_num:] + [0] * padding_num, dtype=np.int32)
                    user_history_mask = np.zeros([self.max_history_num], dtype=bool)
                    user_history_mask[:min(len(history), self.max_history_num)] = True
                    for impression in impressions.strip().split(' '):
                        self.test_indices.append(test_ID)
                        if self.dataset_type == 'MIND-small':
                            self.test_behaviors.append([self.news_ID_dict[impression[:-2]], user_history, user_history_mask])
                        else: # For MIND_large, the test set is not labled
                            self.test_behaviors.append([self.news_ID_dict[impression], user_history, user_history_mask])
                else:
                    for impression in impressions.strip().split(' '):
                        self.test_indices.append(test_ID)
                        if self.dataset_type == 'MIND-small':
                            self.test_behaviors.append([self.news_ID_dict[impression[:-2]], np.zeros([self.max_history_num], dtype=np.int32), np.zeros([self.max_history_num], dtype=bool)])
                        else: # For MIND_large, the test set is not labled
                            self.test_behaviors.append([self.news_ID_dict[impression], np.zeros([self.max_history_num], dtype=np.int32), np.zeros([self.max_history_num], dtype=bool)])
