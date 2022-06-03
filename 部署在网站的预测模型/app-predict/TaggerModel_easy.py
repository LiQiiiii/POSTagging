import torch
import torch.nn as nn
import torch.nn.functional as F
from rnn_easy import RNNEncoder
from char_encoder import CharEncoder

 
class POSTagger(nn.Module):
	def __init__(self, vocab, config, wd_embedding_weights=None, char_embedding_weights=None):
		super(POSTagger, self).__init__()

		self.config = config
		self.bidirectional = True
		self.wd_embedding_size = wd_embedding_weights.shape[1]
		self.wd_embedding = nn.Embedding.from_pretrained(torch.from_numpy(wd_embedding_weights))
		self.wd_embedding.weight.requires_grad = False

		self.rnn_encoder = RNNEncoder(
			input_size=self.wd_embedding_size + self.config.char_hidden_size,   #character-ltsm层的输出会和word_embeddings层输出的词向量进行拼接，拼接后的结果作为新的词向量输入到tag-lstm中进行序列的标注
			hidden_size=self.config.hidden_size,
			num_layers=self.config.nb_layers,
			dropout=self.config.drop_rate,
			bidirectional=self.bidirectional,
			batch_first=True
		)

		self.char_encoder = CharEncoder(
			config=config,
			char_embedding_weights=char_embedding_weights
		)

		num_directions = 2 if self.bidirectional else 1
		# self.hidden2pos = nn.Linear(config.hidden_size, vocab.pos_size)
		self.hidden2pos = nn.Linear(num_directions * config.hidden_size, vocab.pos_size)
		self.embed_dropout = nn.Dropout(config.drop_embed_rate)
		self.linear_dropout = nn.Dropout(config.drop_rate)

	def forward(self, inputs, chars, seq_lens):
		# print('input shape: ', inputs.shape)  # (batch_size, seq_len)
		# print('chars shape:', chars.shape)  # (batch_size*seq_len, wd_len)

		char_representation = self.char_encoder(chars)
		wd_embed = self.wd_embedding(inputs)
		embed = torch.cat((wd_embed, char_representation), dim=2)

		if self.training:  # 预测时要置为False
			embed = self.embed_dropout(embed)

		rnn_out, hidden = self.rnn_encoder(embed, seq_lens)  # (batch_size, seq_len, hidden_size)

		# if self.bidirectional:
		# 	rnn_out = rnn_out[:, :, :self.config.hidden_size] + rnn_out[:, :, self.config.hidden_size:]

		if self.training:
			rnn_out = self.linear_dropout(rnn_out)

		pos_space = self.hidden2pos(rnn_out)  # (batch_size, seq_len, pos_size)
		pos_space = pos_space.reshape(-1, pos_space.size(-1))  # (batch_size * seq_len, pos_size)
		pos_score = F.log_softmax(pos_space, dim=1)  # (batch_size * seq_len, pos_size)  	对行作归一化

		return pos_score
