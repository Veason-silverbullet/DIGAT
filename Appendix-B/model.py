from config import Config
import torch.nn as nn
import newsEncoders
import userEncoders


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        if config.model == 'NRMS':
            self.news_encoder = newsEncoders.NRMS_NewsEncoder(config)
        elif config.model == 'NRMS-SA':
            self.news_encoder = newsEncoders.SA_NRMS_NewsEncoder(config)
        else:
            raise Exception(config.model + ' is not implemented')
        self.user_encoder = userEncoders.NRMS_UserEncoder(self.news_encoder, config)
        self.model_name = config.model
        self.news_embedding_dim = self.news_encoder.news_embedding_dim

    def initialize(self):
        self.news_encoder.initialize()
        self.user_encoder.initialize()

    def forward(self, user_title_text, user_title_mask, user_history_mask, \
                      news_title_text, news_title_mask, augmented_news_title_text, augmented_news_title_mask):
        news_num = news_title_text.size(1)
        news_representation = self.news_encoder(news_title_text, news_title_mask, augmented_news_title_text, augmented_news_title_mask) # [batch_size, 1 + negative_sample_num, news_embedding_dim]
        user_representation = self.user_encoder(user_title_text, user_title_mask, user_history_mask)                                    # [batch_size, news_embedding_dim]
        user_representation = user_representation.unsqueeze(dim=1).repeat(1, news_num, 1)                                               # [batch_size, 1 + negative_sample_num, news_embedding_dim]
        logits = (user_representation * news_representation).sum(dim=2) # dot-product
        return logits
