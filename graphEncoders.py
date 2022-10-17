import math
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ScaledDotProductAttention
from torch_scatter import scatter_sum, scatter_softmax


class GraphEncoder(nn.Module):
    def __init__(self, config: Config, news_embedding_dim: int):
        super(GraphEncoder, self).__init__()
        self.news_graph_size = config.news_graph_size
        self.user_graph_size = config.max_history_num + config.category_num
        self.max_history_num = config.max_history_num
        self.category_num = config.category_num + 1
        self.news_embedding_dim = news_embedding_dim
        self.graph_depth = config.graph_depth
        self.attention_scalar = math.sqrt(float(self.news_embedding_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.topic_node_embedding = nn.Parameter(torch.zeros([config.category_num, self.news_embedding_dim]))
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.dropout__ = nn.Dropout(p=config.dropout_rate/2)
        self.device = torch.device('cuda')

    def initialize(self):
        nn.init.zeros_(self.topic_node_embedding)

    # Input
    # news_graph_embeddings     : [batch_size, news_graph_size, news_embedding_dim]
    # news_graph                : [batch_size, news_graph_size, news_graph_size]
    # news_graph_mask           : [batch_size, news_graph_size]
    # user_news_embedding       : [batch_size, max_history_num, news_embedding_dim]
    # user_graph                : [batch_size, user_graph_size, user_graph_size]
    # user_category_mask        : [batch_size, category_num + 1]
    # user_category_indices     : [batch_size, max_history_num]
    # Output
    # news_graph_representation : [batch_size, news_embedding_dim]
    # user_graph_representation : [batch_size, news_embedding_dim]
    def forward(self, news_graph_embeddings, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices):
        raise Exception('Function forward must be implemented at sub-class')

    def inference(self, news_graph_embeddings, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices, news_graph_context):
        raise Exception('Function inference must be implemented at sub-class')


class DIGAT(GraphEncoder):
    def __init__(self, config: Config, news_embedding_dim: int):
        super(DIGAT, self).__init__(config, news_embedding_dim)
        # compute_news_graph_context
        self.candidate_attention = ScaledDotProductAttention(self.news_embedding_dim, self.news_embedding_dim, self.news_embedding_dim)
        self.news_graph_W = nn.Linear(self.news_embedding_dim * 2, self.news_embedding_dim, bias=True)

        # compute_user_graph_context
        self.user_news_K = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False)
        self.user_news_Q = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.featureAffine = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.userAttention = ScaledDotProductAttention(self.news_embedding_dim, self.news_embedding_dim, self.news_embedding_dim)

        # compute_news_graph_embeddings
        self.news_graph_attention_W = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.news_graph_attention_ffn1 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False) for i in range(self.graph_depth)])
        self.news_graph_attention_ffn2 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False) for i in range(self.graph_depth)])
        self.news_graph_attention_ffn3 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.news_graph_attention_a = nn.ModuleList([nn.Linear(self.news_embedding_dim, 1, bias=False) for i in range(self.graph_depth)])

        # compute_user_graph_embeddings
        self.user_graph_attention_W = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.user_graph_attention_ffn1 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False) for i in range(self.graph_depth)])
        self.user_graph_attention_ffn2 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False) for i in range(self.graph_depth)])
        self.user_graph_attention_ffn3 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.user_graph_attention_a = nn.ModuleList([nn.Linear(self.news_embedding_dim, 1, bias=False) for i in range(self.graph_depth)])


    def initialize(self):
        super().initialize()
        for i in range(self.graph_depth):
            nn.init.xavier_uniform_(self.news_graph_attention_W[i].weight)
            nn.init.zeros_(self.news_graph_attention_W[i].bias)
            nn.init.xavier_uniform_(self.news_graph_attention_a[i].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
            nn.init.xavier_uniform_(self.user_graph_attention_W[i].weight)
            nn.init.zeros_(self.user_graph_attention_W[i].bias)
            nn.init.xavier_uniform_(self.user_graph_attention_a[i].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
            nn.init.xavier_uniform_(self.news_graph_attention_ffn1[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.news_graph_attention_ffn2[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.news_graph_attention_ffn3[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.news_graph_attention_ffn3[i].bias)
            nn.init.xavier_uniform_(self.user_graph_attention_ffn1[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.user_graph_attention_ffn2[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.user_graph_attention_ffn3[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.user_graph_attention_ffn3[i].bias)
        self.candidate_attention.initialize()
        nn.init.xavier_uniform_(self.news_graph_W.weight)
        nn.init.zeros_(self.news_graph_W.bias)
        nn.init.xavier_uniform_(self.user_news_K.weight)
        nn.init.xavier_uniform_(self.user_news_Q.weight)
        nn.init.zeros_(self.user_news_Q.bias)
        nn.init.xavier_uniform_(self.featureAffine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.featureAffine.bias)
        self.userAttention.initialize()


    # Input
    # news_graph_embeddings : [batch_size, news_graph_size, news_embedding_dim]
    # news_graph_mask       : [batch_size, news_graph_size]
    # Output
    # news_graph_context    : [batch_size, news_embedding_dim]
    def compute_news_graph_context(self, news_graph_embeddings, news_graph_mask):
        local_graph_context = news_graph_embeddings.select(1, 0)                                                               # [batch_size, news_embedding_dim]
        global_graph_context = self.candidate_attention(news_graph_embeddings, local_graph_context, mask=news_graph_mask)      # [batch_size, news_embedding_dim]
        gate = torch.sigmoid(self.dropout__(self.news_graph_W(torch.cat([local_graph_context, global_graph_context], dim=1)))) # [batch_size, news_embedding_dim]
        news_graph_context = gate * local_graph_context + (1 - gate) * global_graph_context                                    # [batch_size, news_embedding_dim]
        return news_graph_context

    # Input
    # user_graph_embeddings : [batch_size, max_history_num + category_num, news_embedding_dim]
    # user_category_mask    : [batch_size, category_num + 1]
    # user_category_indices : [batch_size, max_history_num]
    # news_graph_context    : [batch_size, news_embedding_dim]
    # Output
    # user_graph_context    : [batch_size, news_embedding_dim]
    def compute_user_graph_context(self, user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context):
        user_history_embeddings = user_graph_embeddings[:, :self.max_history_num, :]                                              # [batch_size, max_history_num, news_embedding_dim]
        # 1. topic-level attention
        K = self.user_news_K(user_history_embeddings)                                                                             # [batch_size, max_history_num, news_embedding_dim]
        Q = self.user_news_Q(news_graph_context).unsqueeze(dim=2)                                                                 # [batch_size, news_embedding_dim, 1]
        a = torch.bmm(K, Q).squeeze(dim=2) / self.attention_scalar                                                                # [batch_size, max_history_num]
        alpha = scatter_softmax(a, user_category_indices, 1).unsqueeze(dim=2)                                                     # [batch_size, max_history_num, 1]
        topic_embeddings = scatter_sum(alpha * user_history_embeddings, user_category_indices, dim=1, dim_size=self.category_num) # [batch_size, category_num, news_embedding_dim]
        topic_embeddings = self.dropout(F.relu(self.featureAffine(topic_embeddings), inplace=True) + topic_embeddings)            # [batch_size, category_num, news_embedding_dim]
        # 2. user-level attention
        user_graph_context = self.userAttention(topic_embeddings, news_graph_context, mask=user_category_mask)                    # [batch_size, news_embedding_dim]
        return user_graph_context

    # Input
    # index
    # news_graph_embeddings : [batch_size, news_graph_size, news_embedding_dim]
    # news_graph            : [batch_size, news_graph_size, news_graph_size]
    # user_graph_context    : [batch_size, news_embedding_dim]
    # Output
    # news_graph_embeddings : [batch_size, news_graph_size, news_embedding_dim]
    def compute_news_graph_embeddings(self, index, news_graph_embeddings, news_graph, user_graph_context):
        batch_size = news_graph_embeddings.size(0)
        news_graph_embeddings = self.dropout__(news_graph_embeddings)
        h = self.news_graph_attention_W[index](news_graph_embeddings)                                                    # [batch_size, news_graph_size, news_embedding_dim]
        K1 = self.news_graph_attention_ffn1[index](news_graph_embeddings).unsqueeze(dim=1)                               # [batch_size, 1, news_graph_size, news_embedding_dim]
        K2 = self.news_graph_attention_ffn2[index](news_graph_embeddings).unsqueeze(dim=2)                               # [batch_size, news_graph_size, 1, news_embedding_dim]
        K3 = self.news_graph_attention_ffn3[index](user_graph_context).view([batch_size, 1, 1, self.news_embedding_dim]) # [batch_size, 1, 1, news_embedding_dim]
        a = self.news_graph_attention_a[index](F.relu(K3 + K1 + K2, inplace=True)).squeeze(dim=3)                        # [batch_size, news_graph_size, news_graph_size]
        e = self.leaky_relu(a)                                                                                           # [batch_size, news_graph_size, news_graph_size]
        alpha = self.dropout_(F.softmax(e.masked_fill(news_graph == 0, -1e9), dim=2))                                    # [batch_size, news_graph_size, news_graph_size]
        _news_graph_embeddings = F.relu(torch.bmm(alpha, h), inplace=True) + news_graph_embeddings                       # [batch_size, news_graph_size, news_embedding_dim]
        return _news_graph_embeddings

    # Input
    # index
    # user_graph_embeddings : [batch_size, user_graph_size, news_embedding_dim]
    # user_graph            : [batch_size, user_graph_size, user_graph_size]
    # news_graph_context    : [batch_size, news_embedding_dim]
    # Output
    # user_graph_embeddings : [batch_size, user_graph_size, news_embedding_dim]
    def compute_user_graph_embeddings(self, index, user_graph_embeddings, user_graph, news_graph_context):
        batch_size = user_graph_embeddings.size(0)
        user_graph_embeddings = self.dropout__(user_graph_embeddings)
        h = self.user_graph_attention_W[index](user_graph_embeddings)                                                    # [batch_size, user_graph_size, news_embedding_dim]
        K1 = self.user_graph_attention_ffn1[index](user_graph_embeddings).unsqueeze(dim=1)                               # [batch_size, 1, user_graph_size, news_embedding_dim]
        K2 = self.user_graph_attention_ffn2[index](user_graph_embeddings).unsqueeze(dim=2)                               # [batch_size, user_graph_size, 1, news_embedding_dim]
        K3 = self.user_graph_attention_ffn3[index](news_graph_context).view([batch_size, 1, 1, self.news_embedding_dim]) # [batch_size, 1, 1, news_embedding_dim]
        a = self.user_graph_attention_a[index](F.relu(K3 + K1 + K2, inplace=True)).squeeze(dim=3)                        # [batch_size, user_graph_size, user_graph_size]
        e = self.leaky_relu(a)                                                                                           # [batch_size, user_graph_size, user_graph_size]
        alpha = self.dropout_(F.softmax(e.masked_fill(user_graph == 0, -1e9), dim=2))                                    # [batch_size, user_graph_size, user_graph_size]
        _user_graph_embeddings = F.relu(torch.bmm(alpha, h), inplace=True) + user_graph_embeddings                       # [batch_size, user_graph_size, news_embedding_dim]
        return _user_graph_embeddings


    def forward(self, news_graph_embeddings, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices):
        batch_size = news_graph_embeddings.size(0)
        user_graph_embeddings = torch.cat([user_news_embedding, self.dropout__(self.topic_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1))], dim=1)              # [batch_size, user_graph_size, news_embedding_dim]
        news_graph_context = self.compute_news_graph_context(news_graph_embeddings, news_graph_mask)                                                                        # [batch_size, news_embedding_dim]
        user_graph_context = self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context)                          # [batch_size, news_embedding_dim]
        for i in range(self.graph_depth):
            news_graph_embeddings = self.compute_news_graph_embeddings(i, news_graph_embeddings, news_graph, user_graph_context)                                            # [batch_size, news_graph_size, news_embedding_dim]
            user_graph_embeddings = self.compute_user_graph_embeddings(i, user_graph_embeddings, user_graph, news_graph_context)                                            # [batch_size, user_graph_size, news_embedding_dim]
            news_graph_context = news_graph_context + self.compute_news_graph_context(news_graph_embeddings, news_graph_mask)                                               # [batch_size, news_embedding_dim]
            user_graph_context = user_graph_context + self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context) # [batch_size, news_embedding_dim]
        return news_graph_context, user_graph_context

    def inference(self, news_graph_embeddings, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices, news_graph_context):
        batch_size = news_graph_embeddings.size(0)
        user_graph_embeddings = torch.cat([user_news_embedding, self.topic_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1)], dim=1)                              # [batch_size, user_graph_size, news_embedding_dim]
        user_graph_context = self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context)                          # [batch_size, news_embedding_dim]
        for i in range(self.graph_depth):
            news_graph_embeddings = self.compute_news_graph_embeddings(i, news_graph_embeddings, news_graph, user_graph_context)                                            # [batch_size, news_graph_size, news_embedding_dim]
            user_graph_embeddings = self.compute_user_graph_embeddings(i, user_graph_embeddings, user_graph, news_graph_context)                                            # [batch_size, user_graph_size, news_embedding_dim]
            news_graph_context = news_graph_context + self.compute_news_graph_context(news_graph_embeddings, news_graph_mask)                                               # [batch_size, news_embedding_dim]
            user_graph_context = user_graph_context + self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context) # [batch_size, news_embedding_dim]
        return news_graph_context, user_graph_context


class wo_SA(GraphEncoder):
    def __init__(self, config: Config, news_embedding_dim: int):
        super(wo_SA, self).__init__(config, news_embedding_dim)
        # compute_user_graph_context
        self.user_news_K = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False)
        self.user_news_Q = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.featureAffine = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.userAttention = ScaledDotProductAttention(self.news_embedding_dim, self.news_embedding_dim, self.news_embedding_dim)

        # compute_user_graph_embeddings
        self.user_graph_attention_W = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.user_graph_attention_ffn1 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False) for i in range(self.graph_depth)])
        self.user_graph_attention_ffn2 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False) for i in range(self.graph_depth)])
        self.user_graph_attention_ffn3 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.user_graph_attention_a = nn.ModuleList([nn.Linear(self.news_embedding_dim, 1, bias=False) for i in range(self.graph_depth)])


    def initialize(self):
        super().initialize()
        for i in range(self.graph_depth):
            nn.init.xavier_uniform_(self.user_graph_attention_W[i].weight)
            nn.init.zeros_(self.user_graph_attention_W[i].bias)
            nn.init.xavier_uniform_(self.user_graph_attention_ffn1[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.user_graph_attention_ffn2[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.user_graph_attention_ffn3[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.user_graph_attention_ffn3[i].bias)
            nn.init.xavier_uniform_(self.user_graph_attention_a[i].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        nn.init.xavier_uniform_(self.user_news_K.weight)
        nn.init.xavier_uniform_(self.user_news_Q.weight)
        nn.init.zeros_(self.user_news_Q.bias)
        nn.init.xavier_uniform_(self.featureAffine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.featureAffine.bias)
        self.userAttention.initialize()

    # Input
    # user_graph_embeddings : [batch_size, max_history_num + category_num, news_embedding_dim]
    # user_category_mask    : [batch_size, category_num + 1]
    # user_category_indices : [batch_size, max_history_num]
    # news_graph_context    : [batch_size, news_embedding_dim]
    # Output
    # user_graph_context    : [batch_size, news_embedding_dim]
    def compute_user_graph_context(self, user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context):
        user_history_embeddings = user_graph_embeddings[:, :self.max_history_num, :]                                              # [batch_size, max_history_num, news_embedding_dim]
        # 1. topic-level attention
        K = self.user_news_K(user_history_embeddings)                                                                             # [batch_size, max_history_num, news_embedding_dim]
        Q = self.user_news_Q(news_graph_context).unsqueeze(dim=2)                                                                 # [batch_size, news_embedding_dim, 1]
        a = torch.bmm(K, Q).squeeze(dim=2) / self.attention_scalar                                                                # [batch_size, max_history_num]
        alpha = scatter_softmax(a, user_category_indices, 1).unsqueeze(dim=2)                                                     # [batch_size, max_history_num, 1]
        topic_embeddings = scatter_sum(alpha * user_history_embeddings, user_category_indices, dim=1, dim_size=self.category_num) # [batch_size, category_num, news_embedding_dim]
        topic_embeddings = self.dropout(F.relu(self.featureAffine(topic_embeddings), inplace=True) + topic_embeddings)            # [batch_size, category_num, news_embedding_dim]
        # 2. user-level attention
        user_graph_context = self.userAttention(topic_embeddings, news_graph_context, mask=user_category_mask)                    # [batch_size, news_embedding_dim]
        return user_graph_context

    # Input
    # index
    # user_graph_embeddings : [batch_size, user_graph_size, news_embedding_dim]
    # user_graph            : [batch_size, user_graph_size, user_graph_size]
    # news_graph_context    : [batch_size, news_embedding_dim]
    # Output
    # user_graph_embeddings : [batch_size, user_graph_size, news_embedding_dim]
    def compute_user_graph_embeddings(self, index, user_graph_embeddings, user_graph, news_graph_context):
        batch_size = user_graph_embeddings.size(0)
        user_graph_embeddings = self.dropout__(user_graph_embeddings)
        h = self.user_graph_attention_W[index](user_graph_embeddings)                                                    # [batch_size, user_graph_size, news_embedding_dim]
        K1 = self.user_graph_attention_ffn1[index](user_graph_embeddings).unsqueeze(dim=1)                               # [batch_size, 1, user_graph_size, news_embedding_dim]
        K2 = self.user_graph_attention_ffn2[index](user_graph_embeddings).unsqueeze(dim=2)                               # [batch_size, user_graph_size, 1, news_embedding_dim]
        K3 = self.user_graph_attention_ffn3[index](news_graph_context).view([batch_size, 1, 1, self.news_embedding_dim]) # [batch_size, 1, 1, news_embedding_dim]
        a = self.user_graph_attention_a[index](F.relu(K3 + K1 + K2, inplace=True)).squeeze(dim=3)                        # [batch_size, user_graph_size, user_graph_size]
        e = self.leaky_relu(a)                                                                                           # [batch_size, user_graph_size, user_graph_size]
        alpha = self.dropout_(F.softmax(e.masked_fill(user_graph == 0, -1e9), dim=2))                                    # [batch_size, user_graph_size, user_graph_size]
        _user_graph_embeddings = F.relu(torch.bmm(alpha, h), inplace=True) + user_graph_embeddings                       # [batch_size, user_graph_size, news_embedding_dim]
        return _user_graph_embeddings


    def forward(self, news_graph_embeddings, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices):
        batch_size = news_graph_embeddings.size(0)
        user_graph_embeddings = torch.cat([user_news_embedding, self.dropout__(self.topic_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1))], dim=1)       # [batch_size, user_graph_size, news_embedding_dim]
        single_candidate_news_representation = news_graph_embeddings.select(dim=1, index=0)                                                                          # [batch_size, news_embedding_dim]
        for i in range(self.graph_depth):
            user_graph_embeddings = self.compute_user_graph_embeddings(i, user_graph_embeddings, user_graph, single_candidate_news_representation)                   # [batch_size, user_graph_size, news_embedding_dim]
        user_graph_context = self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, single_candidate_news_representation) # [batch_size, news_embedding_dim]
        return single_candidate_news_representation, user_graph_context

    def inference(self, news_graph_embeddings, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices, news_graph_context):
        batch_size = news_graph_embeddings.size(0)
        user_graph_embeddings = torch.cat([user_news_embedding, self.topic_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1)], dim=1)                       # [batch_size, user_graph_size, news_embedding_dim]
        single_candidate_news_representation = news_graph_embeddings.select(dim=1, index=0)                                                                          # [batch_size, news_embedding_dim]
        for i in range(self.graph_depth):
            user_graph_embeddings = self.compute_user_graph_embeddings(i, user_graph_embeddings, user_graph, single_candidate_news_representation)                   # [batch_size, user_graph_size, news_embedding_dim]
        user_graph_context = self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, single_candidate_news_representation) # [batch_size, news_embedding_dim]
        return single_candidate_news_representation, user_graph_context


class Seq_SA(GraphEncoder):
    def __init__(self, config: Config, news_embedding_dim: int):
        super(Seq_SA, self).__init__(config, news_embedding_dim)
        # compute_news_sequence_context
        self.candidate_attention = ScaledDotProductAttention(self.news_embedding_dim, self.news_embedding_dim, self.news_embedding_dim)
        self.news_graph_W = nn.Linear(self.news_embedding_dim * 2, self.news_embedding_dim, bias=True)

        # compute_user_graph_context
        self.user_news_K = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False)
        self.user_news_Q = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.featureAffine = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.userAttention = ScaledDotProductAttention(self.news_embedding_dim, self.news_embedding_dim, self.news_embedding_dim)

        # compute_user_graph_embeddings
        self.user_graph_attention_W = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.user_graph_attention_ffn1 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False) for i in range(self.graph_depth)])
        self.user_graph_attention_ffn2 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False) for i in range(self.graph_depth)])
        self.user_graph_attention_ffn3 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.user_graph_attention_a = nn.ModuleList([nn.Linear(self.news_embedding_dim, 1, bias=False) for i in range(self.graph_depth)])


    def initialize(self):
        super().initialize()
        for i in range(self.graph_depth):
            nn.init.xavier_uniform_(self.user_graph_attention_W[i].weight)
            nn.init.zeros_(self.user_graph_attention_W[i].bias)
            nn.init.xavier_uniform_(self.user_graph_attention_ffn1[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.user_graph_attention_ffn2[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.user_graph_attention_ffn3[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.user_graph_attention_ffn3[i].bias)
            nn.init.xavier_uniform_(self.user_graph_attention_a[i].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        self.candidate_attention.initialize()
        nn.init.xavier_uniform_(self.news_graph_W.weight)
        nn.init.zeros_(self.news_graph_W.bias)
        nn.init.xavier_uniform_(self.user_news_K.weight)
        nn.init.xavier_uniform_(self.user_news_Q.weight)
        nn.init.zeros_(self.user_news_Q.bias)
        nn.init.xavier_uniform_(self.featureAffine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.featureAffine.bias)
        self.userAttention.initialize()


    # Input
    # news_graph_embeddings : [batch_size, news_graph_size, news_embedding_dim]
    # news_graph_mask       : [batch_size, news_graph_size]
    # Output
    # news_graph_context    : [batch_size, news_embedding_dim]
    def compute_news_sequence_context(self, news_graph_embeddings, news_graph_mask):
        local_graph_context = news_graph_embeddings.select(1, 0)                                                               # [batch_size, news_embedding_dim]
        global_graph_context = self.candidate_attention(news_graph_embeddings, local_graph_context, mask=news_graph_mask)      # [batch_size, news_embedding_dim]
        gate = torch.sigmoid(self.dropout__(self.news_graph_W(torch.cat([local_graph_context, global_graph_context], dim=1)))) # [batch_size, news_embedding_dim]
        news_graph_context = gate * local_graph_context + (1 - gate) * global_graph_context                                    # [batch_size, news_embedding_dim]
        return news_graph_context

    # Input
    # user_graph_embeddings : [batch_size, max_history_num + category_num, news_embedding_dim]
    # user_category_mask    : [batch_size, category_num + 1]
    # user_category_indices : [batch_size, max_history_num]
    # news_graph_context    : [batch_size, news_embedding_dim]
    # Output
    # user_graph_context    : [batch_size, news_embedding_dim]
    def compute_user_graph_context(self, user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context):
        user_history_embeddings = user_graph_embeddings[:, :self.max_history_num, :]                                              # [batch_size, max_history_num, news_embedding_dim]
        # 1. topic-level attention
        K = self.user_news_K(user_history_embeddings)                                                                             # [batch_size, max_history_num, news_embedding_dim]
        Q = self.user_news_Q(news_graph_context).unsqueeze(dim=2)                                                                 # [batch_size, news_embedding_dim, 1]
        a = torch.bmm(K, Q).squeeze(dim=2) / self.attention_scalar                                                                # [batch_size, max_history_num]
        alpha = scatter_softmax(a, user_category_indices, 1).unsqueeze(dim=2)                                                     # [batch_size, max_history_num, 1]
        topic_embeddings = scatter_sum(alpha * user_history_embeddings, user_category_indices, dim=1, dim_size=self.category_num) # [batch_size, category_num, news_embedding_dim]
        topic_embeddings = self.dropout(F.relu(self.featureAffine(topic_embeddings), inplace=True) + topic_embeddings)            # [batch_size, category_num, news_embedding_dim]
        # 2. user-level attention
        user_graph_context = self.userAttention(topic_embeddings, news_graph_context, mask=user_category_mask)                    # [batch_size, news_embedding_dim]
        return user_graph_context

    # Input
    # index
    # user_graph_embeddings : [batch_size, user_graph_size, news_embedding_dim]
    # user_graph            : [batch_size, user_graph_size, user_graph_size]
    # news_graph_context    : [batch_size, news_embedding_dim]
    # Output
    # user_graph_embeddings : [batch_size, user_graph_size, news_embedding_dim]
    def compute_user_graph_embeddings(self, index, user_graph_embeddings, user_graph, news_graph_context):
        batch_size = user_graph_embeddings.size(0)
        user_graph_embeddings = self.dropout__(user_graph_embeddings)
        h = self.user_graph_attention_W[index](user_graph_embeddings)                                                    # [batch_size, user_graph_size, news_embedding_dim]
        K1 = self.user_graph_attention_ffn1[index](user_graph_embeddings).unsqueeze(dim=1)                               # [batch_size, 1, user_graph_size, news_embedding_dim]
        K2 = self.user_graph_attention_ffn2[index](user_graph_embeddings).unsqueeze(dim=2)                               # [batch_size, user_graph_size, 1, news_embedding_dim]
        K3 = self.user_graph_attention_ffn3[index](news_graph_context).view([batch_size, 1, 1, self.news_embedding_dim]) # [batch_size, 1, 1, news_embedding_dim]
        a = self.user_graph_attention_a[index](F.relu(K3 + K1 + K2, inplace=True)).squeeze(dim=3)                        # [batch_size, user_graph_size, user_graph_size]
        e = self.leaky_relu(a)                                                                                           # [batch_size, user_graph_size, user_graph_size]
        alpha = self.dropout_(F.softmax(e.masked_fill(user_graph == 0, -1e9), dim=2))                                    # [batch_size, user_graph_size, user_graph_size]
        _user_graph_embeddings = F.relu(torch.bmm(alpha, h), inplace=True) + user_graph_embeddings                       # [batch_size, user_graph_size, news_embedding_dim]
        return _user_graph_embeddings


    def forward(self, news_sequence_embeddings, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices):
        batch_size = news_sequence_embeddings.size(0)
        user_graph_embeddings = torch.cat([user_news_embedding, self.dropout__(self.topic_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1))], dim=1)                 # [batch_size, user_graph_size, news_embedding_dim]
        news_sequence_context = self.compute_news_sequence_context(news_sequence_embeddings, news_graph_mask)                                                                  # [batch_size, news_embedding_dim]
        user_graph_context = self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_sequence_context)                          # [batch_size, news_embedding_dim]
        for i in range(self.graph_depth):
            user_graph_embeddings = self.compute_user_graph_embeddings(i, user_graph_embeddings, user_graph, news_sequence_context)                                            # [batch_size, user_graph_size, news_embedding_dim]
            user_graph_context = user_graph_context + self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_sequence_context) # [batch_size, news_embedding_dim]
        return news_sequence_context, user_graph_context

    def inference(self, news_graph_embeddings, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices, news_sequence_context):
        batch_size = news_graph_embeddings.size(0)
        user_graph_embeddings = torch.cat([user_news_embedding, self.topic_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1)], dim=1)                                 # [batch_size, user_graph_size, news_embedding_dim]
        user_graph_context = self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_sequence_context)                          # [batch_size, news_embedding_dim]
        for i in range(self.graph_depth):
            user_graph_embeddings = self.compute_user_graph_embeddings(i, user_graph_embeddings, user_graph, news_sequence_context)                                            # [batch_size, user_graph_size, news_embedding_dim]
            user_graph_context = user_graph_context + self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_sequence_context) # [batch_size, news_embedding_dim]
        return news_sequence_context, user_graph_context


class wo_interaction(GraphEncoder):
    def __init__(self, config: Config, news_embedding_dim: int):
        super(wo_interaction, self).__init__(config, news_embedding_dim)
        # compute_news_graph_context
        self.candidate_attention = ScaledDotProductAttention(self.news_embedding_dim, self.news_embedding_dim, self.news_embedding_dim)
        self.news_graph_W = nn.Linear(self.news_embedding_dim * 2, self.news_embedding_dim, bias=True)

        # compute_user_graph_context
        self.user_news_K = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False)
        self.user_news_Q = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.featureAffine = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.userAttention = ScaledDotProductAttention(self.news_embedding_dim, self.news_embedding_dim, self.news_embedding_dim)

        # compute_news_graph_embeddings
        self.news_graph_attention_W = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.news_graph_attention_a1 = nn.ModuleList([nn.Linear(self.news_embedding_dim, 1, bias=False) for i in range(self.graph_depth)])
        self.news_graph_attention_a2 = nn.ModuleList([nn.Linear(self.news_embedding_dim, 1, bias=False) for i in range(self.graph_depth)])

        # compute_user_graph_embeddings
        self.user_graph_attention_W = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.user_graph_attention_a1 = nn.ModuleList([nn.Linear(self.news_embedding_dim, 1, bias=False) for i in range(self.graph_depth)])
        self.user_graph_attention_a2 = nn.ModuleList([nn.Linear(self.news_embedding_dim, 1, bias=False) for i in range(self.graph_depth)])


    def initialize(self):
        super().initialize()
        for i in range(self.graph_depth):
            nn.init.xavier_uniform_(self.news_graph_attention_W[i].weight)
            nn.init.zeros_(self.news_graph_attention_W[i].bias)
            nn.init.xavier_uniform_(self.news_graph_attention_a1[i].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
            nn.init.xavier_uniform_(self.news_graph_attention_a2[i].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
            nn.init.xavier_uniform_(self.user_graph_attention_W[i].weight)
            nn.init.zeros_(self.user_graph_attention_W[i].bias)
            nn.init.xavier_uniform_(self.user_graph_attention_a1[i].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
            nn.init.xavier_uniform_(self.user_graph_attention_a2[i].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        self.candidate_attention.initialize()
        nn.init.xavier_uniform_(self.news_graph_W.weight)
        nn.init.zeros_(self.news_graph_W.bias)
        nn.init.xavier_uniform_(self.user_news_K.weight)
        nn.init.xavier_uniform_(self.user_news_Q.weight)
        nn.init.zeros_(self.user_news_Q.bias)
        nn.init.xavier_uniform_(self.featureAffine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.featureAffine.bias)
        self.userAttention.initialize()


    # Input
    # news_graph_embeddings : [batch_size, news_graph_size, news_embedding_dim]
    # news_graph_mask       : [batch_size, news_graph_size]
    # Output
    # news_graph_context    : [batch_size, news_embedding_dim]
    def compute_news_graph_context(self, news_graph_embeddings, news_graph_mask):
        local_graph_context = news_graph_embeddings.select(1, 0)                                                               # [batch_size, news_embedding_dim]
        global_graph_context = self.candidate_attention(news_graph_embeddings, local_graph_context, mask=news_graph_mask)      # [batch_size, news_embedding_dim]
        gate = torch.sigmoid(self.dropout__(self.news_graph_W(torch.cat([local_graph_context, global_graph_context], dim=1)))) # [batch_size, news_embedding_dim]
        news_graph_context = gate * local_graph_context + (1 - gate) * global_graph_context                                    # [batch_size, news_embedding_dim]
        return news_graph_context

    # Input
    # user_graph_embeddings : [batch_size, max_history_num + category_num, news_embedding_dim]
    # user_category_mask    : [batch_size, category_num + 1]
    # user_category_indices : [batch_size, max_history_num]
    # news_graph_context    : [batch_size, news_embedding_dim]
    # Output
    # user_graph_context    : [batch_size, news_embedding_dim]
    def compute_user_graph_context(self, user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context):
        user_history_embeddings = user_graph_embeddings[:, :self.max_history_num, :]                                              # [batch_size, max_history_num, news_embedding_dim]
        # 1. topic-level attention
        K = self.user_news_K(user_history_embeddings)                                                                             # [batch_size, max_history_num, news_embedding_dim]
        Q = self.user_news_Q(news_graph_context).unsqueeze(dim=2)                                                                 # [batch_size, news_embedding_dim, 1]
        a = torch.bmm(K, Q).squeeze(dim=2) / self.attention_scalar                                                                # [batch_size, max_history_num]
        alpha = scatter_softmax(a, user_category_indices, 1).unsqueeze(dim=2)                                                     # [batch_size, max_history_num, 1]
        topic_embeddings = scatter_sum(alpha * user_history_embeddings, user_category_indices, dim=1, dim_size=self.category_num) # [batch_size, category_num, news_embedding_dim]
        topic_embeddings = self.dropout(F.relu(self.featureAffine(topic_embeddings), inplace=True) + topic_embeddings)            # [batch_size, category_num, news_embedding_dim]
        # 2. user-level attention
        user_graph_context = self.userAttention(topic_embeddings, news_graph_context, mask=user_category_mask)                    # [batch_size, news_embedding_dim]
        return user_graph_context

    # Input
    # index
    # news_graph_embeddings : [batch_size, news_graph_size, news_embedding_dim]
    # news_graph            : [batch_size, news_graph_size, news_graph_size]
    # Output
    # news_graph_embeddings : [batch_size, news_graph_size, news_embedding_dim]
    def compute_news_graph_embeddings(self, index, news_graph_embeddings, news_graph):
        batch_size = news_graph_embeddings.size(0)
        news_graph_embeddings = self.dropout__(news_graph_embeddings)
        h = self.news_graph_attention_W[index](news_graph_embeddings)                              # [batch_size, news_graph_size, news_embedding_dim]
        a1 = self.news_graph_attention_a1[index](h).view([batch_size, 1, self.news_graph_size])    # [batch_size, 1, news_graph_size]
        a2 = self.news_graph_attention_a2[index](h)                                                # [batch_size, news_graph_size, 1]
        e = self.leaky_relu(a1 + a2)                                                               # [batch_size, user_graph_size, user_graph_size]
        alpha = self.dropout_(F.softmax(e.masked_fill(news_graph == 0, -1e9), dim=2))              # [batch_size, news_graph_size, news_graph_size]
        _news_graph_embeddings = F.relu(torch.bmm(alpha, h), inplace=True) + news_graph_embeddings # [batch_size, news_graph_size, news_embedding_dim]
        return _news_graph_embeddings

    # Input
    # index
    # user_graph_embeddings : [batch_size, user_graph_size, news_embedding_dim]
    # user_graph            : [batch_size, user_graph_size, user_graph_size]
    # Output
    # user_graph_embeddings : [batch_size, user_graph_size, news_embedding_dim]
    def compute_user_graph_embeddings(self, index, user_graph_embeddings, user_graph):
        batch_size = user_graph_embeddings.size(0)
        user_graph_embeddings = self.dropout__(user_graph_embeddings)
        h = self.user_graph_attention_W[index](user_graph_embeddings)                              # [batch_size, user_graph_size, news_embedding_dim]
        a1 = self.user_graph_attention_a1[index](h).view([batch_size, 1, self.user_graph_size])    # [batch_size, 1, user_graph_size]
        a2 = self.user_graph_attention_a2[index](h)                                                # [batch_size, user_graph_size, 1]
        e = self.leaky_relu(a1 + a2)                                                               # [batch_size, user_graph_size, user_graph_size]
        alpha = self.dropout_(F.softmax(e.masked_fill(user_graph == 0, -1e9), dim=2))              # [batch_size, user_graph_size, user_graph_size]
        _user_graph_embeddings = F.relu(torch.bmm(alpha, h), inplace=True) + user_graph_embeddings # [batch_size, user_graph_size, news_embedding_dim]
        return _user_graph_embeddings


    def forward(self, news_graph_embeddings, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices):
        batch_size = news_graph_embeddings.size(0)
        user_graph_embeddings = torch.cat([user_news_embedding, self.dropout__(self.topic_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1))], dim=1)              # [batch_size, user_graph_size, news_embedding_dim]
        news_graph_context = self.compute_news_graph_context(news_graph_embeddings, news_graph_mask)                                                                        # [batch_size, news_embedding_dim]
        user_graph_context = self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context)                          # [batch_size, news_embedding_dim]
        for i in range(self.graph_depth):
            # replace news graph embedding update layer with vanilla GAT update layer
            news_graph_embeddings = self.compute_news_graph_embeddings(i, news_graph_embeddings, news_graph)                                                                # [batch_size, news_graph_size, news_embedding_dim]
            # replace user graph embedding update layer with vanilla GAT update layer
            user_graph_embeddings = self.compute_user_graph_embeddings(i, user_graph_embeddings, user_graph)                                                                # [batch_size, user_graph_size, news_embedding_dim]
            news_graph_context = news_graph_context + self.compute_news_graph_context(news_graph_embeddings, news_graph_mask)                                               # [batch_size, news_embedding_dim]
            user_graph_context = user_graph_context + self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context) # [batch_size, news_embedding_dim]
        return news_graph_context, user_graph_context

    def inference(self, news_graph_embeddings, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices, news_graph_context):
        batch_size = news_graph_embeddings.size(0)
        user_graph_embeddings = torch.cat([user_news_embedding, self.topic_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1)], dim=1)                              # [batch_size, user_graph_size, news_embedding_dim]
        user_graph_context = self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context)                          # [batch_size, news_embedding_dim]
        for i in range(self.graph_depth):
            # replace news graph embedding update layer with vanilla GAT update layer
            news_graph_embeddings = self.compute_news_graph_embeddings(i, news_graph_embeddings, news_graph)                                                                # [batch_size, news_graph_size, news_embedding_dim]
            # replace user graph embedding update layer with vanilla GAT update layer
            user_graph_embeddings = self.compute_user_graph_embeddings(i, user_graph_embeddings, user_graph)                                                                # [batch_size, user_graph_size, news_embedding_dim]
            news_graph_context = news_graph_context + self.compute_news_graph_context(news_graph_embeddings, news_graph_mask)                                               # [batch_size, news_embedding_dim]
            user_graph_context = user_graph_context + self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context) # [batch_size, news_embedding_dim]
        return news_graph_context, user_graph_context


class News_graph_wo_inter(GraphEncoder):
    def __init__(self, config: Config, news_embedding_dim: int):
        super(News_graph_wo_inter, self).__init__(config, news_embedding_dim)
        # compute_news_graph_context
        self.candidate_attention = ScaledDotProductAttention(self.news_embedding_dim, self.news_embedding_dim, self.news_embedding_dim)
        self.news_graph_W = nn.Linear(self.news_embedding_dim * 2, self.news_embedding_dim, bias=True)

        # compute_user_graph_context
        self.user_news_K = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False)
        self.user_news_Q = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.featureAffine = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.userAttention = ScaledDotProductAttention(self.news_embedding_dim, self.news_embedding_dim, self.news_embedding_dim)

        # compute_news_graph_embeddings
        self.news_graph_attention_W = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.news_graph_attention_a1 = nn.ModuleList([nn.Linear(self.news_embedding_dim, 1, bias=False) for i in range(self.graph_depth)])
        self.news_graph_attention_a2 = nn.ModuleList([nn.Linear(self.news_embedding_dim, 1, bias=False) for i in range(self.graph_depth)])

        # compute_user_graph_embeddings
        self.user_graph_attention_W = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.user_graph_attention_ffn1 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False) for i in range(self.graph_depth)])
        self.user_graph_attention_ffn2 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False) for i in range(self.graph_depth)])
        self.user_graph_attention_ffn3 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.user_graph_attention_a = nn.ModuleList([nn.Linear(self.news_embedding_dim, 1, bias=False) for i in range(self.graph_depth)])


    def initialize(self):
        super().initialize()
        for i in range(self.graph_depth):
            nn.init.xavier_uniform_(self.news_graph_attention_W[i].weight)
            nn.init.zeros_(self.news_graph_attention_W[i].bias)
            nn.init.xavier_uniform_(self.news_graph_attention_a1[i].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
            nn.init.xavier_uniform_(self.news_graph_attention_a2[i].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
            nn.init.xavier_uniform_(self.user_graph_attention_W[i].weight)
            nn.init.zeros_(self.user_graph_attention_W[i].bias)
            nn.init.xavier_uniform_(self.user_graph_attention_ffn1[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.user_graph_attention_ffn2[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.user_graph_attention_ffn3[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.user_graph_attention_ffn3[i].bias)
            nn.init.xavier_uniform_(self.user_graph_attention_a[i].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        self.candidate_attention.initialize()
        nn.init.xavier_uniform_(self.news_graph_W.weight)
        nn.init.zeros_(self.news_graph_W.bias)
        nn.init.xavier_uniform_(self.user_news_K.weight)
        nn.init.xavier_uniform_(self.user_news_Q.weight)
        nn.init.zeros_(self.user_news_Q.bias)
        nn.init.xavier_uniform_(self.featureAffine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.featureAffine.bias)
        self.userAttention.initialize()


    # Input
    # news_graph_embeddings : [batch_size, news_graph_size, news_embedding_dim]
    # news_graph_mask       : [batch_size, news_graph_size]
    # Output
    # news_graph_context    : [batch_size, news_embedding_dim]
    def compute_news_graph_context(self, news_graph_embeddings, news_graph_mask):
        local_graph_context = news_graph_embeddings.select(1, 0)                                                               # [batch_size, news_embedding_dim]
        global_graph_context = self.candidate_attention(news_graph_embeddings, local_graph_context, mask=news_graph_mask)      # [batch_size, news_embedding_dim]
        gate = torch.sigmoid(self.dropout__(self.news_graph_W(torch.cat([local_graph_context, global_graph_context], dim=1)))) # [batch_size, news_embedding_dim]
        news_graph_context = gate * local_graph_context + (1 - gate) * global_graph_context                                    # [batch_size, news_embedding_dim]
        return news_graph_context

    # Input
    # user_graph_embeddings : [batch_size, max_history_num + category_num, news_embedding_dim]
    # user_category_mask    : [batch_size, category_num + 1]
    # user_category_indices : [batch_size, max_history_num]
    # news_graph_context    : [batch_size, news_embedding_dim]
    # Output
    # user_graph_context    : [batch_size, news_embedding_dim]
    def compute_user_graph_context(self, user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context):
        user_history_embeddings = user_graph_embeddings[:, :self.max_history_num, :]                                              # [batch_size, max_history_num, news_embedding_dim]
        # 1. topic-level attention
        K = self.user_news_K(user_history_embeddings)                                                                             # [batch_size, max_history_num, news_embedding_dim]
        Q = self.user_news_Q(news_graph_context).unsqueeze(dim=2)                                                                 # [batch_size, news_embedding_dim, 1]
        a = torch.bmm(K, Q).squeeze(dim=2) / self.attention_scalar                                                                # [batch_size, max_history_num]
        alpha = scatter_softmax(a, user_category_indices, 1).unsqueeze(dim=2)                                                     # [batch_size, max_history_num, 1]
        topic_embeddings = scatter_sum(alpha * user_history_embeddings, user_category_indices, dim=1, dim_size=self.category_num) # [batch_size, category_num, news_embedding_dim]
        topic_embeddings = self.dropout(F.relu(self.featureAffine(topic_embeddings), inplace=True) + topic_embeddings)            # [batch_size, category_num, news_embedding_dim]
        # 2. user-level attention
        user_graph_context = self.userAttention(topic_embeddings, news_graph_context, mask=user_category_mask)                    # [batch_size, news_embedding_dim]
        return user_graph_context

    # Input
    # index
    # news_graph_embeddings : [batch_size, news_graph_size, news_embedding_dim]
    # news_graph            : [batch_size, news_graph_size, news_graph_size]
    # Output
    # news_graph_embeddings : [batch_size, news_graph_size, news_embedding_dim]
    def compute_news_graph_embeddings(self, index, news_graph_embeddings, news_graph):
        batch_size = news_graph_embeddings.size(0)
        news_graph_embeddings = self.dropout__(news_graph_embeddings)
        h = self.news_graph_attention_W[index](news_graph_embeddings)                              # [batch_size, news_graph_size, news_embedding_dim]
        a1 = self.news_graph_attention_a1[index](h).view([batch_size, 1, self.news_graph_size])    # [batch_size, 1, news_graph_size]
        a2 = self.news_graph_attention_a2[index](h)                                                # [batch_size, news_graph_size, 1]
        e = self.leaky_relu(a1 + a2)                                                               # [batch_size, user_graph_size, user_graph_size]
        alpha = self.dropout_(F.softmax(e.masked_fill(news_graph == 0, -1e9), dim=2))              # [batch_size, news_graph_size, news_graph_size]
        _news_graph_embeddings = F.relu(torch.bmm(alpha, h), inplace=True) + news_graph_embeddings # [batch_size, news_graph_size, news_embedding_dim]
        return _news_graph_embeddings

    # Input
    # index
    # user_graph_embeddings : [batch_size, user_graph_size, news_embedding_dim]
    # user_graph            : [batch_size, user_graph_size, user_graph_size]
    # news_graph_context    : [batch_size, news_embedding_dim]
    # Output
    # user_graph_embeddings : [batch_size, user_graph_size, news_embedding_dim]
    def compute_user_graph_embeddings(self, index, user_graph_embeddings, user_graph, news_graph_context):
        batch_size = user_graph_embeddings.size(0)
        user_graph_embeddings = self.dropout__(user_graph_embeddings)
        h = self.user_graph_attention_W[index](user_graph_embeddings)                                                    # [batch_size, user_graph_size, news_embedding_dim]
        K1 = self.user_graph_attention_ffn1[index](user_graph_embeddings).unsqueeze(dim=1)                               # [batch_size, 1, user_graph_size, news_embedding_dim]
        K2 = self.user_graph_attention_ffn2[index](user_graph_embeddings).unsqueeze(dim=2)                               # [batch_size, user_graph_size, 1, news_embedding_dim]
        K3 = self.user_graph_attention_ffn3[index](news_graph_context).view([batch_size, 1, 1, self.news_embedding_dim]) # [batch_size, 1, 1, news_embedding_dim]
        a = self.user_graph_attention_a[index](F.relu(K3 + K1 + K2, inplace=True)).squeeze(dim=3)                        # [batch_size, user_graph_size, user_graph_size]
        e = self.leaky_relu(a)                                                                                           # [batch_size, user_graph_size, user_graph_size]
        alpha = self.dropout_(F.softmax(e.masked_fill(user_graph == 0, -1e9), dim=2))                                    # [batch_size, user_graph_size, user_graph_size]
        _user_graph_embeddings = F.relu(torch.bmm(alpha, h), inplace=True) + user_graph_embeddings                       # [batch_size, user_graph_size, news_embedding_dim]
        return _user_graph_embeddings


    def forward(self, news_graph_embeddings, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices):
        batch_size = news_graph_embeddings.size(0)
        user_graph_embeddings = torch.cat([user_news_embedding, self.dropout__(self.topic_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1))], dim=1)              # [batch_size, user_graph_size, news_embedding_dim]
        news_graph_context = self.compute_news_graph_context(news_graph_embeddings, news_graph_mask)                                                                        # [batch_size, news_embedding_dim]
        user_graph_context = self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context)                          # [batch_size, news_embedding_dim]
        for i in range(self.graph_depth):
            # replace news graph embedding update layer with vanilla GAT update layer
            news_graph_embeddings = self.compute_news_graph_embeddings(i, news_graph_embeddings, news_graph)                                                                # [batch_size, news_graph_size, news_embedding_dim]
            user_graph_embeddings = self.compute_user_graph_embeddings(i, user_graph_embeddings, user_graph, news_graph_context)                                            # [batch_size, user_graph_size, news_embedding_dim]
            news_graph_context = news_graph_context + self.compute_news_graph_context(news_graph_embeddings, news_graph_mask)                                               # [batch_size, news_embedding_dim]
            user_graph_context = user_graph_context + self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context) # [batch_size, news_embedding_dim]
        return news_graph_context, user_graph_context

    def inference(self, news_graph_embeddings, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices, news_graph_context):
        batch_size = news_graph_embeddings.size(0)
        user_graph_embeddings = torch.cat([user_news_embedding, self.topic_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1)], dim=1)                              # [batch_size, user_graph_size, news_embedding_dim]
        user_graph_context = self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context)                          # [batch_size, news_embedding_dim]
        for i in range(self.graph_depth):
            # replace news graph embedding update layer with vanilla GAT update layer
            news_graph_embeddings = self.compute_news_graph_embeddings(i, news_graph_embeddings, news_graph)                                                                # [batch_size, news_graph_size, news_embedding_dim]
            user_graph_embeddings = self.compute_user_graph_embeddings(i, user_graph_embeddings, user_graph, news_graph_context)                                            # [batch_size, user_graph_size, news_embedding_dim]
            news_graph_context = news_graph_context + self.compute_news_graph_context(news_graph_embeddings, news_graph_mask)                                               # [batch_size, news_embedding_dim]
            user_graph_context = user_graph_context + self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context) # [batch_size, news_embedding_dim]
        return news_graph_context, user_graph_context


class User_graph_wo_inter(GraphEncoder):
    def __init__(self, config: Config, news_embedding_dim: int):
        super(User_graph_wo_inter, self).__init__(config, news_embedding_dim)
        # compute_news_graph_context
        self.candidate_attention = ScaledDotProductAttention(self.news_embedding_dim, self.news_embedding_dim, self.news_embedding_dim)
        self.news_graph_W = nn.Linear(self.news_embedding_dim * 2, self.news_embedding_dim, bias=True)

        # compute_user_graph_context
        self.user_news_K = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False)
        self.user_news_Q = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.featureAffine = nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True)
        self.userAttention = ScaledDotProductAttention(self.news_embedding_dim, self.news_embedding_dim, self.news_embedding_dim)

        # compute_news_graph_embeddings
        self.news_graph_attention_W = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.news_graph_attention_ffn1 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False) for i in range(self.graph_depth)])
        self.news_graph_attention_ffn2 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=False) for i in range(self.graph_depth)])
        self.news_graph_attention_ffn3 = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.news_graph_attention_a = nn.ModuleList([nn.Linear(self.news_embedding_dim, 1, bias=False) for i in range(self.graph_depth)])

        # compute_user_graph_embeddings
        self.user_graph_attention_W = nn.ModuleList([nn.Linear(self.news_embedding_dim, self.news_embedding_dim, bias=True) for i in range(self.graph_depth)])
        self.user_graph_attention_a1 = nn.ModuleList([nn.Linear(self.news_embedding_dim, 1, bias=False) for i in range(self.graph_depth)])
        self.user_graph_attention_a2 = nn.ModuleList([nn.Linear(self.news_embedding_dim, 1, bias=False) for i in range(self.graph_depth)])


    def initialize(self):
        super().initialize()
        for i in range(self.graph_depth):
            nn.init.xavier_uniform_(self.news_graph_attention_W[i].weight)
            nn.init.zeros_(self.news_graph_attention_W[i].bias)
            nn.init.xavier_uniform_(self.news_graph_attention_ffn1[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.news_graph_attention_ffn2[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.news_graph_attention_ffn3[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.news_graph_attention_ffn3[i].bias)
            nn.init.xavier_uniform_(self.news_graph_attention_a[i].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
            nn.init.xavier_uniform_(self.user_graph_attention_W[i].weight)
            nn.init.zeros_(self.user_graph_attention_W[i].bias)
            nn.init.xavier_uniform_(self.user_graph_attention_a1[i].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
            nn.init.xavier_uniform_(self.user_graph_attention_a2[i].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        self.candidate_attention.initialize()
        nn.init.xavier_uniform_(self.news_graph_W.weight)
        nn.init.zeros_(self.news_graph_W.bias)
        nn.init.xavier_uniform_(self.user_news_K.weight)
        nn.init.xavier_uniform_(self.user_news_Q.weight)
        nn.init.zeros_(self.user_news_Q.bias)
        nn.init.xavier_uniform_(self.featureAffine.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.featureAffine.bias)
        self.userAttention.initialize()


    # Input
    # news_graph_embeddings : [batch_size, news_graph_size, news_embedding_dim]
    # news_graph_mask       : [batch_size, news_graph_size]
    # Output
    # news_graph_context    : [batch_size, news_embedding_dim]
    def compute_news_graph_context(self, news_graph_embeddings, news_graph_mask):
        local_graph_context = news_graph_embeddings.select(1, 0)                                                               # [batch_size, news_embedding_dim]
        global_graph_context = self.candidate_attention(news_graph_embeddings, local_graph_context, mask=news_graph_mask)      # [batch_size, news_embedding_dim]
        gate = torch.sigmoid(self.dropout__(self.news_graph_W(torch.cat([local_graph_context, global_graph_context], dim=1)))) # [batch_size, news_embedding_dim]
        news_graph_context = gate * local_graph_context + (1 - gate) * global_graph_context                                    # [batch_size, news_embedding_dim]
        return news_graph_context

    # Input
    # user_graph_embeddings : [batch_size, max_history_num + category_num, news_embedding_dim]
    # user_category_mask    : [batch_size, category_num + 1]
    # user_category_indices : [batch_size, max_history_num]
    # news_graph_context    : [batch_size, news_embedding_dim]
    # Output
    # user_graph_context    : [batch_size, news_embedding_dim]
    def compute_user_graph_context(self, user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context):
        user_history_embeddings = user_graph_embeddings[:, :self.max_history_num, :]                                              # [batch_size, max_history_num, news_embedding_dim]
        # 1. topic-level attention
        K = self.user_news_K(user_history_embeddings)                                                                             # [batch_size, max_history_num, news_embedding_dim]
        Q = self.user_news_Q(news_graph_context).unsqueeze(dim=2)                                                                 # [batch_size, news_embedding_dim, 1]
        a = torch.bmm(K, Q).squeeze(dim=2) / self.attention_scalar                                                                # [batch_size, max_history_num]
        alpha = scatter_softmax(a, user_category_indices, 1).unsqueeze(dim=2)                                                     # [batch_size, max_history_num, 1]
        topic_embeddings = scatter_sum(alpha * user_history_embeddings, user_category_indices, dim=1, dim_size=self.category_num) # [batch_size, category_num, news_embedding_dim]
        topic_embeddings = self.dropout(F.relu(self.featureAffine(topic_embeddings), inplace=True) + topic_embeddings)            # [batch_size, category_num, news_embedding_dim]
        # 2. user-level attention
        user_graph_context = self.userAttention(topic_embeddings, news_graph_context, mask=user_category_mask)                    # [batch_size, news_embedding_dim]
        return user_graph_context

    # Input
    # index
    # news_graph_embeddings : [batch_size, news_graph_size, news_embedding_dim]
    # news_graph            : [batch_size, news_graph_size, news_graph_size]
    # user_graph_context    : [batch_size, news_embedding_dim]
    # Output
    # news_graph_embeddings : [batch_size, news_graph_size, news_embedding_dim]
    def compute_news_graph_embeddings(self, index, news_graph_embeddings, news_graph, user_graph_context):
        batch_size = news_graph_embeddings.size(0)
        news_graph_embeddings = self.dropout__(news_graph_embeddings)
        h = self.news_graph_attention_W[index](news_graph_embeddings)                                                    # [batch_size, news_graph_size, news_embedding_dim]
        K1 = self.news_graph_attention_ffn1[index](news_graph_embeddings).unsqueeze(dim=1)                               # [batch_size, 1, news_graph_size, news_embedding_dim]
        K2 = self.news_graph_attention_ffn2[index](news_graph_embeddings).unsqueeze(dim=2)                               # [batch_size, news_graph_size, 1, news_embedding_dim]
        K3 = self.news_graph_attention_ffn3[index](user_graph_context).view([batch_size, 1, 1, self.news_embedding_dim]) # [batch_size, 1, 1, news_embedding_dim]
        a = self.news_graph_attention_a[index](F.relu(K3 + K1 + K2, inplace=True)).squeeze(dim=3)                        # [batch_size, news_graph_size, news_graph_size]
        e = self.leaky_relu(a)                                                                                           # [batch_size, news_graph_size, news_graph_size]
        alpha = self.dropout_(F.softmax(e.masked_fill(news_graph == 0, -1e9), dim=2))                                    # [batch_size, news_graph_size, news_graph_size]
        _news_graph_embeddings = F.relu(torch.bmm(alpha, h), inplace=True) + news_graph_embeddings                       # [batch_size, news_graph_size, news_embedding_dim]
        return _news_graph_embeddings

    # Input
    # index
    # user_graph_embeddings : [batch_size, user_graph_size, news_embedding_dim]
    # user_graph            : [batch_size, user_graph_size, user_graph_size]
    # Output
    # user_graph_embeddings : [batch_size, user_graph_size, news_embedding_dim]
    def compute_user_graph_embeddings(self, index, user_graph_embeddings, user_graph):
        batch_size = user_graph_embeddings.size(0)
        user_graph_embeddings = self.dropout__(user_graph_embeddings)
        h = self.user_graph_attention_W[index](user_graph_embeddings)                              # [batch_size, user_graph_size, news_embedding_dim]
        a1 = self.user_graph_attention_a1[index](h).view([batch_size, 1, self.user_graph_size])    # [batch_size, 1, user_graph_size]
        a2 = self.user_graph_attention_a2[index](h)                                                # [batch_size, user_graph_size, 1]
        e = self.leaky_relu(a1 + a2)                                                               # [batch_size, user_graph_size, user_graph_size]
        alpha = self.dropout_(F.softmax(e.masked_fill(user_graph == 0, -1e9), dim=2))              # [batch_size, user_graph_size, user_graph_size]
        _user_graph_embeddings = F.relu(torch.bmm(alpha, h), inplace=True) + user_graph_embeddings # [batch_size, user_graph_size, news_embedding_dim]
        return _user_graph_embeddings


    def forward(self, news_graph_embeddings, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices):
        batch_size = news_graph_embeddings.size(0)
        user_graph_embeddings = torch.cat([user_news_embedding, self.dropout__(self.topic_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1))], dim=1)              # [batch_size, user_graph_size, news_embedding_dim]
        news_graph_context = self.compute_news_graph_context(news_graph_embeddings, news_graph_mask)                                                                        # [batch_size, news_embedding_dim]
        user_graph_context = self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context)                          # [batch_size, news_embedding_dim]
        for i in range(self.graph_depth):
            news_graph_embeddings = self.compute_news_graph_embeddings(i, news_graph_embeddings, news_graph, user_graph_context)                                            # [batch_size, news_graph_size, news_embedding_dim]
            # replace user graph embedding update layer with vanilla GAT update layer
            user_graph_embeddings = self.compute_user_graph_embeddings(i, user_graph_embeddings, user_graph)                                                                # [batch_size, user_graph_size, news_embedding_dim]
            news_graph_context = news_graph_context + self.compute_news_graph_context(news_graph_embeddings, news_graph_mask)                                               # [batch_size, news_embedding_dim]
            user_graph_context = user_graph_context + self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context) # [batch_size, news_embedding_dim]
        return news_graph_context, user_graph_context

    def inference(self, news_graph_embeddings, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices, news_graph_context):
        batch_size = news_graph_embeddings.size(0)
        user_graph_embeddings = torch.cat([user_news_embedding, self.topic_node_embedding.unsqueeze(dim=0).expand(batch_size, -1, -1)], dim=1)                              # [batch_size, user_graph_size, news_embedding_dim]
        user_graph_context = self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context)                          # [batch_size, news_embedding_dim]
        for i in range(self.graph_depth):
            news_graph_embeddings = self.compute_news_graph_embeddings(i, news_graph_embeddings, news_graph, user_graph_context)                                            # [batch_size, news_graph_size, news_embedding_dim]
            # replace user graph embedding update layer with vanilla GAT update layer
            user_graph_embeddings = self.compute_user_graph_embeddings(i, user_graph_embeddings, user_graph)                                                                # [batch_size, user_graph_size, news_embedding_dim]
            news_graph_context = news_graph_context + self.compute_news_graph_context(news_graph_embeddings, news_graph_mask)                                               # [batch_size, news_embedding_dim]
            user_graph_context = user_graph_context + self.compute_user_graph_context(user_graph_embeddings, user_category_mask, user_category_indices, news_graph_context) # [batch_size, news_embedding_dim]
        return news_graph_context, user_graph_context
