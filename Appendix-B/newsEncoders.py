import pickle
from config import Config
import torch
import torch.nn as nn
from layers import MultiHeadAttention, Attention, ScaledDotProductAttention


class NewsEncoder(nn.Module):
    def __init__(self, config: Config):
        super(NewsEncoder, self).__init__()
        self.max_sentence_length = config.max_title_length
        self.word_embedding_dim = config.word_embedding_dim
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        with open('word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + str(config.max_title_length) + '-' + config.dataset + '.pkl', 'rb') as word_embedding_f:
            self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate/2, inplace=True)
        self.augmented_news_num = config.augmented_news_num

    def initialize(self):
        pass

    # Input
    # title_text           : [batch_size, news_num, max_title_length]
    # title_mask           : [batch_size, news_num, max_title_length]
    # augmented_title_text : [batch_size, news_num, augmented_news_num, max_title_length]
    # augmented_title_mask : [batch_size, news_num, augmented_news_num, max_title_length]
    # Output
    # news_representation  : [batch_size, news_num, news_embedding_dim]
    def forward(self, title_text, title_mask, augmented_news_title_text, augmented_news_title_mask):
        raise Exception('Function forward must be implemented at sub-class')


class NRMS_NewsEncoder(NewsEncoder):
    def __init__(self, config: Config):
        super(NRMS_NewsEncoder, self).__init__(config)
        self.news_embedding_dim = config.head_num * config.head_dim
        self.multiheadAttention = MultiHeadAttention(config.head_num, config.word_embedding_dim, config.max_title_length, config.max_title_length, config.head_dim, config.head_dim)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)

    def initialize(self):
        super().initialize()
        self.multiheadAttention.initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask, augmented_news_title_text=None, augmented_news_title_mask=None):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                                   # [batch_news_num, max_sentence_length]

        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim])          # [batch_news_num, max_sentence_length, word_embedding_dim]
        # 2. multi-head self-attention
        c = self.dropout(self.multiheadAttention(w, w, w, mask))                                                                             # [batch_news_num, max_sentence_length, news_embedding_dim]
        # 3. attention layer
        news_representation = self.attention(c, mask=mask).view([batch_size, news_num, self.news_embedding_dim])                             # [batch_size, news_num, news_embedding_dim]
        return news_representation


class SA_NRMS_NewsEncoder(NewsEncoder):
    def __init__(self, config: Config):
        super(SA_NRMS_NewsEncoder, self).__init__(config)
        self.news_embedding_dim = config.head_num * config.head_dim
        self.multiheadAttention = MultiHeadAttention(config.head_num, config.word_embedding_dim, config.max_title_length, config.max_title_length, config.head_dim, config.head_dim)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)
        self.SA_attention = ScaledDotProductAttention(self.news_embedding_dim, self.news_embedding_dim, self.news_embedding_dim)
        self.SA_transformation = nn.Linear(in_features=self.news_embedding_dim*2, out_features=self.news_embedding_dim, bias=True)

    def initialize(self):
        super().initialize()
        self.multiheadAttention.initialize()
        self.attention.initialize()
        self.SA_attention.initialize()
        nn.init.xavier_uniform_(self.SA_transformation.weight)
        nn.init.zeros_(self.SA_transformation.bias)

    def forward(self, title_text, title_mask, augmented_news_title_text=None, augmented_news_title_mask=None):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                                                                 # [batch_news_num, max_sentence_length]
        if augmented_news_title_text is not None:
            augmented_batch_news_num = batch_size * news_num * self.augmented_news_num
            augmented_news_title_text = augmented_news_title_text.view([augmented_batch_news_num, self.max_sentence_length])                                               # [augmented_batch_news_num, max_sentence_length, word_embedding_dim]
            augmented_news_title_mask = augmented_news_title_mask.view([augmented_batch_news_num, self.max_sentence_length])                                               # [augmented_batch_news_num, max_sentence_length, word_embedding_dim]

        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim])                                        # [batch_news_num, max_sentence_length, word_embedding_dim]
        # 2. multi-head self-attention
        c = self.dropout(self.multiheadAttention(w, w, w, mask))                                                                                                           # [batch_news_num, max_sentence_length, news_embedding_dim]
        # 3. attention layer
        news_representation = self.attention(c, mask=mask).view([batch_size, news_num, self.news_embedding_dim])                                                           # [batch_size, news_num, news_embedding_dim]

        if augmented_news_title_text is not None:
            _w = self.dropout(self.word_embedding(augmented_news_title_text)).view([augmented_batch_news_num, self.max_sentence_length, self.word_embedding_dim])          # [augmented_batch_news_num, max_sentence_length, word_embedding_dim]
            _c = self.dropout(self.multiheadAttention(_w, _w, _w, augmented_news_title_mask))                                                                              # [augmented_batch_news_num, max_sentence_length, news_embedding_dim]
            _news_representation = self.attention(_c, mask=augmented_news_title_mask).view([batch_news_num, self.augmented_news_num, self.news_embedding_dim])             # [batch_news_num, augmented_news_num, news_embedding_dim]
            original_news_representation = news_representation.view([batch_news_num, self.news_embedding_dim])                                                             # [batch_news_num, news_embedding_dim]
            augmented_news_representation = self.SA_attention(_news_representation, original_news_representation)                                                          # [batch_news_num, news_embedding_dim]
            gate = torch.sigmoid(self.dropout_(self.SA_transformation(torch.cat([original_news_representation, augmented_news_representation], dim=1))))                   # [batch_news_num, news_embedding_dim]
            news_representation = (gate * original_news_representation + (1 - gate) * augmented_news_representation).view([batch_size, news_num, self.news_embedding_dim]) # [batch_news_num, news_num, news_embedding_dim]
        return news_representation
