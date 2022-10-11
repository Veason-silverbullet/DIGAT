import pickle
from config import Config
import torch.nn as nn
import torch.nn.functional as F
from layers import Conv1D, MultiHeadAttention, Attention


class NewsEncoder(nn.Module):
    def __init__(self, config: Config):
        super(NewsEncoder, self).__init__()
        self.word_embedding_dim = config.word_embedding_dim
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        with open('word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + str(config.max_title_length) + '-' + str(config.dataset) + '.pkl', 'rb') as word_embedding_f:
            self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)

    def initialize(self):
        pass

    # Input
    # title_text          : [batch_size, news_num, max_title_length]
    # title_mask          : [batch_size, news_num, max_title_length]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def forward(self, title_text, title_mask):
        raise Exception('Function forward must be implemented at sub-class')


class CNN(NewsEncoder):
    def __init__(self, config: Config):
        super(CNN, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.news_embedding_dim = config.cnn_kernel_num
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)

    def initialize(self):
        super().initialize()
        self.conv.initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. CNN encoding
        h = self.dropout(self.conv(w.permute(0, 2, 1)).permute(0, 2, 1))                                                            # [batch_size * news_num, max_sentence_length, news_embedding_dim]
        # 3. attention aggregation layer
        news_representation = self.attention(h, mask=mask).view([batch_size, news_num, self.news_embedding_dim])                    # [batch_size, news_num, news_embedding_dim]
        return news_representation


# we implement multihead self-attention (MSA) news encoder following the official Pytorch transformer encoder(https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer)
class MSA(NewsEncoder):
    def __init__(self, config: Config):
        super(MSA, self).__init__(config)
        self.max_sentence_length = config.max_title_length
        self.multiheadSelfattention = MultiHeadAttention(config.MSA_head_num, config.word_embedding_dim, config.max_title_length, config.max_title_length, config.MSA_head_dim, config.MSA_head_dim)
        self.news_embedding_dim = config.MSA_head_num * config.MSA_head_dim
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)

    def initialize(self):
        super().initialize()
        self.multiheadSelfattention.initialize()
        self.attention.initialize()

    def forward(self, title_text, title_mask):
        batch_size = title_text.size(0)
        news_num = title_text.size(1)
        batch_news_num = batch_size * news_num
        mask = title_mask.view([batch_news_num, self.max_sentence_length])                                                          # [batch_size * news_num, max_sentence_length]
        # 1. word embedding
        w = self.dropout(self.word_embedding(title_text)).view([batch_news_num, self.max_sentence_length, self.word_embedding_dim]) # [batch_size * news_num, max_sentence_length, word_embedding_dim]
        # 2. multi-head self-attention
        h = F.relu(self.multiheadSelfattention(w, w, w), inplace=True)                                                              # [batch_size * news_num, max_sentence_length, news_embedding_dim]
        # 3. attention aggregation layer
        news_representation = self.attention(h, mask=mask).view([batch_size, news_num, self.news_embedding_dim])                    # [batch_size, news_num, news_embedding_dim]
        return news_representation
