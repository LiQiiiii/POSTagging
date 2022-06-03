# from collections import Counter, defaultdict
import sys
sys.path.extend(['./', '../', '../../'])
import torch
import numpy as np
from dataloader.Vocab import POSVocab, CharVocab


# 一个Instance对应一行记录
class Instance(object):
	"""
	将word 和 pos 封装成一个实体
	Args:
		object:
	return:
		Instance(word, pos)
	"""
	def __init__(self, words, pos):
		self.words = words  # 保存词序列
		self.pos = pos      # 保存词序列对应的词性序列

	def __str__(self):
		return ' '.join([wd+'_'+p for wd, p in zip(self.words, self.pos)])


# 加载数据集，数据封装成Instance实体
def load_data(corpus_path):	#一个word，一个pos的格式
	"""

	Args:
		corpus_path: 文件路径

	Returns:
		Instance(word, pos)
	"""
	insts = []
	with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as fin:
		for line in fin:
			if(line == "\n"):
				continue
			if(len(line.strip().split('\t')) != 2):
				continue
			words= []
			pos = []
			words.append(line.strip().split('\t')[0])
			pos.append(line.strip().split('\t')[1])
			insts.append(Instance(words, pos))
	return insts			

# 获取batch数据
def get_batch(data, batch_size, shuffle=True):   
	"""
	Args:
		data ： (数据集)
		batch_size : 批处理个数
		shuffle : 是否打乱数据集

	Yields:
		batch_data: 一个batch的数据集
	"""
	if shuffle:
		np.random.shuffle(data)

	num_batch = int(np.ceil(len(data) / float(batch_size)))
	for i in range(num_batch):
		batch_data = data[i*batch_size: (i+1)*batch_size]
		if shuffle:
			np.random.shuffle(batch_data)

		yield batch_data


def create_vocabs(corpus_path):	
	"""

	Args:
		corpus_path ： 词典路径

	Returns:
		CharVocab(char_set) : 生成CharVocab实体
 		PosVocab(pos_set) : 生成PosVocab实体
	"""
	words_set = set()
	char_set = set()
	pos_set = set()
	with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as fin:
		for line in fin:
			if(line == "\n"):
				continue
			if(len(line.strip().split('\t')) != 2):
				continue
			wd, pos = line.strip().split('\t')
			char_set.update([ch.strip() for ch in wd])
			words_set.add(wd)
			pos_set.add(pos)

	return CharVocab(char_set), POSVocab(words_set, pos_set)


def pred_data_variable(insts, vocab, char_vocab):
	"""
	Args:
		insts : Instance(word, pos)
        vocab : 中文词词典
        char_vocab : 字符词典

	Returns:
		wds_ids : 每个词对应的id索引词典
		char_ids : 每个字符所对应的d字符词典
		seq_lens : 序列长度
	"""
	batch_size = len(insts)
	max_seq_len, max_wd_len = 0, 0
	for inst in insts:
		if len(inst.words) > max_seq_len:
			max_seq_len = len(inst.words)
		for wd in inst.words:
			if len(wd) > max_wd_len:
				max_wd_len = len(wd)

	wds_idxs = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
	char_idxs = torch.zeros((batch_size, max_seq_len, max_wd_len), dtype=torch.long)
	seq_lens = torch.zeros(batch_size, )

	for i, inst in enumerate(insts):
		seq_len = len(inst.words)
		seq_lens[i] = seq_len
		for j, wd in enumerate(inst.words):
			char_idxs[i, j, :len(wd)] = torch.tensor(char_vocab.char2idx(wd), dtype=torch.long)
		wds_idxs[i, :seq_len] = torch.tensor(vocab.word2index(inst.words), dtype=torch.long)

	return wds_idxs, char_idxs, seq_lens


def batch_variable_mask_easy(batch_data, vocab, char_vocab):
	"""

	Args:
		batch_data : 批处理数据
        vocab : 中文词词典
        char_vocab : 字符词典
	Returns:
		wds_idxs : 每个词对应的id索引词典
		char_idxs : 每个字符所对应的id字符词典
		pos_idxs : 每个POS所对应的id词性词典
		seq_lens : 序列长度
	"""
	batch_size = len(batch_data)
	max_seq_len, max_wd_len = 0, 0
	for inst in batch_data:
		if len(inst.words) > max_seq_len:
			max_seq_len = len(inst.words)    #每个实体words的长度
		for wd in inst.words:
			if len(wd) > max_wd_len:
				max_wd_len = len(wd)

	wds_idxs = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
	char_idxs = torch.zeros((batch_size, max_seq_len, max_wd_len), dtype=torch.long)
	pos_idxs = torch.zeros(batch_size, max_seq_len, dtype=torch.long).fill_(-1)
	seq_lens = torch.zeros(batch_size, )

	for i, inst in enumerate(batch_data):
		seq_len = len(inst.words)
		seq_lens[i] = seq_len
		for j, wd in enumerate(inst.words):
			char_idxs[i, j, :len(wd)] = torch.tensor(char_vocab.char2idx(wd), dtype=torch.long)
		wds_idxs[i, :seq_len] = torch.tensor(vocab.word2index(inst.words), dtype=torch.long)
		pos_idxs[i, :seq_len] = torch.tensor(vocab.pos2index(inst.pos), dtype=torch.long)

	pos_idxs = pos_idxs.flatten()  # 展平成一维

	return wds_idxs, char_idxs, pos_idxs, seq_lens
