from config import Config
import torch.nn as nn
import newsEncoders
import graphEncoders


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        # News encoder
        if config.news_encoder == 'CNN':
            self.news_encoder = newsEncoders.CNN(config)
        elif config.news_encoder == 'MSA':
            self.news_encoder = newsEncoders.MSA(config)
        else:
            raise Exception(config.news_encoder + ' is not implemented')
        # Graph encoder
        if config.graph_encoder == 'DIGAT':
            self.graph_encoder = graphEncoders.DIGAT(config, self.news_encoder.news_embedding_dim)
        elif config.graph_encoder == 'wo_SA':
            self.graph_encoder = graphEncoders.wo_SA(config, self.news_encoder.news_embedding_dim)
        elif config.graph_encoder == 'Seq_SA':
            self.graph_encoder = graphEncoders.Seq_SA(config, self.news_encoder.news_embedding_dim)
        elif config.graph_encoder == 'wo_interaction':
            self.graph_encoder = graphEncoders.wo_interaction(config, self.news_encoder.news_embedding_dim)
        elif config.graph_encoder == 'news_graph_wo_inter':
            self.graph_encoder = graphEncoders.News_graph_wo_inter(config, self.news_encoder.news_embedding_dim)
        elif config.graph_encoder == 'user_graph_wo_inter':
            self.graph_encoder = graphEncoders.User_graph_wo_inter(config, self.news_encoder.news_embedding_dim)
        else:
            raise Exception(config.graph_encoder + ' is not implemented')
        self.model_name = config.news_encoder + '-' + config.graph_encoder
        self.max_title_length = config.max_title_length
        self.max_history_num = config.max_history_num
        self.category_num = config.category_num + 1
        self.news_embedding_dim = self.news_encoder.news_embedding_dim
        self.representation_dim = self.news_embedding_dim
        self.news_graph_size = config.news_graph_size
        self.user_graph_size = config.max_history_num + config.category_num

    def initialize(self):
        self.news_encoder.initialize()
        self.graph_encoder.initialize()

    # user_title_text       : [batch_size, max_history_num, max_title_length]
    # user_title_mask       : [batch_size, max_history_num, max_title_length]
    # user_graph            : [batch_size, user_graph_size, user_graph_size]
    # user_category_mask    : [batch_size, category_num + 1]
    # user_category_indices : [batch_size, max_history_num]
    # news_title_text       : [batch_size, news_num, news_graph_size, max_title_length]
    # news_title_mask       : [batch_size, news_num, news_graph_size, max_title_length]
    # news_graph            : [batch_size, news_num, news_graph_size, news_graph_size]
    # news_graph_mask       : [batch_size, news_num, news_graph_size]
    def forward(self, user_title_text, user_title_mask, user_graph, user_category_mask, user_category_indices, \
                      news_title_text, news_title_mask, news_graph, news_graph_mask):
        batch_size = news_graph.size(0)
        news_num = news_graph.size(1)
        batch_news_num = batch_size * news_num

        news_title_text = news_title_text.view([batch_news_num, self.news_graph_size, self.max_title_length])
        news_title_mask = news_title_mask.view([batch_news_num, self.news_graph_size, self.max_title_length])
        news_graph = news_graph.view([batch_news_num, self.news_graph_size, self.news_graph_size])
        news_graph_mask = news_graph_mask.view([batch_news_num, self.news_graph_size])
        user_graph = user_graph.unsqueeze(dim=1).expand(-1, news_num, -1, -1).contiguous().view([batch_news_num, self.user_graph_size, self.user_graph_size])
        user_category_mask = user_category_mask.unsqueeze(dim=1).expand(-1, news_num, -1).contiguous().view([batch_news_num, self.category_num])
        user_category_indices = user_category_indices.unsqueeze(dim=1).expand(-1, news_num, -1).contiguous().view([batch_news_num, self.max_history_num])

        candidate_news_embedding = self.news_encoder(news_title_text, news_title_mask)                                  # [batch_news_num, news_graph_size, news_embedding_dim]
        user_news_embedding = self.news_encoder(user_title_text, user_title_mask)                                       # [batch_size, max_history_num, news_embedding_dim]
        user_news_embedding = user_news_embedding.unsqueeze(dim=1).expand(-1, news_num, -1, -1).contiguous()            # [batch_size, news_num, max_history_num, news_embedding_dim]
        user_news_embedding = user_news_embedding.view([batch_news_num, self.max_history_num, self.news_embedding_dim]) # [batch_news_num, max_history_num, news_embedding_dim]

        news_representation, user_representation = self.graph_encoder(candidate_news_embedding, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices)
        news_representation = news_representation.view([batch_size, news_num, self.representation_dim])                 # [batch_size, news_num, representation_dim]
        user_representation = user_representation.view([batch_size, news_num, self.representation_dim])                 # [batch_size, news_num, representation_dim]
        logits = (user_representation * news_representation).sum(dim=2) # dot-product
        return logits

    # user_news_embedding      : [batch_size, max_history_num, news_embedding_dim]
    # user_graph               : [batch_size, user_graph_size, user_graph_size]
    # user_category_mask       : [batch_size, category_num + 1]
    # user_category_indices    : [batch_size, max_history_num]
    # candidate_news_embedding : [batch_size, news_graph_size, news_embedding_dim]
    # news_graph               : [batch_size, news_graph_size, news_graph_size]
    # news_graph_mask          : [batch_size, news_graph_size]
    # c_n0                     : [batch_size, news_embedding_dim]
    def inference(self, user_news_embedding, user_graph, user_category_mask, user_category_indices, candidate_news_embedding, news_graph, news_graph_mask, c_n0):
        news_representation, user_representation = self.graph_encoder.inference(candidate_news_embedding, news_graph, news_graph_mask, user_news_embedding, user_graph, user_category_mask, user_category_indices, c_n0)
        logits = (user_representation * news_representation).sum(dim=1) # dot-product
        return logits
