from config import Config
import torch
import torch.nn as nn
from layers import MultiHeadAttention, Attention
from newsEncoders import NewsEncoder


class UserEncoder(nn.Module):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(UserEncoder, self).__init__()
        self.news_embedding_dim = news_encoder.news_embedding_dim
        self.news_encoder = news_encoder
        self.max_history_num = config.max_history_num
        self.device = torch.device('cuda')

    # Input
    # user_title_text     : [batch_size, max_history_num, max_title_length]
    # user_title_mask     : [batch_size, max_history_num, max_title_length]
    # user_history_mask   : [batch_size, max_history_num]
    # Output
    # user_representation : [batch_size, news_embedding_dim]
    def forward(self, user_ID, user_title_text, user_title_mask, user_history_mask):
        raise Exception('Function forward must be implemented at sub-class')

    # Input
    # history_embedding   : [batch_size, max_history_num, news_embedding_dim]
    # user_history_mask   : [batch_size, max_history_num]
    # Output
    # user_representation : [batch_size, news_embedding_dim]
    def encode(self, user_ID, user_title_text, user_title_mask, user_history_mask):
        raise Exception('Function encode must be implemented at sub-class')


class NRMS_UserEncoder(UserEncoder):
    def __init__(self, news_encoder: NewsEncoder, config: Config):
        super(NRMS_UserEncoder, self).__init__(news_encoder, config)
        self.multiheadAttention = MultiHeadAttention(config.head_num, self.news_embedding_dim, config.max_history_num, config.max_history_num, config.head_dim, config.head_dim)
        self.attention = Attention(self.news_embedding_dim, config.attention_dim)

    def initialize(self):
        self.multiheadAttention.initialize()
        self.attention.initialize()

    def encode(self, history_embedding, user_history_mask):
        h = self.multiheadAttention(history_embedding, history_embedding, history_embedding, user_history_mask) # [batch_size, max_history_num, head_num * head_dim]
        user_representation = self.attention(h)                                                                 # [batch_size, news_embedding_dim]
        return user_representation

    def forward(self, user_title_text, user_title_mask, user_history_mask):
        history_embedding = self.news_encoder(user_title_text, user_title_mask) # [batch_size, max_history_num, news_embedding_dim]
        user_representation = self.encode(history_embedding, user_history_mask) # [batch_size, news_embedding_dim]
        return user_representation
