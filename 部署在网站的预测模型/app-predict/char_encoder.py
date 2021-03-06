import torch
import torch.nn as nn
import torch.nn.functional as F

"""
字符卷积神经网络
令vi为第i个词的词向量，vi的维度为d维。当句子词语数为L
卷积神经网络的滑动窗口大小为k时，落入第j个(j<=L-1)滑动
窗口的词向量依次为vj, vj-1, ..., vj+l-1，可以将他们表达为窗口向量。
对于每个窗口向量，用卷积核W进行卷积运算得到当前窗口特征 
使用Sigmod激活函数加速收敛，再用最大池化操作来最大化词语特征表示。
"""
class CharEncoder(nn.Module):
    def __init__(self, config, char_embedding_weights):
        super(CharEncoder, self).__init__()

        self.char_hidden_size = config.char_hidden_size
        self.vocab_size, self.char_embedding_size = char_embedding_weights.shape
        
        # self.char_embedding = nn.Embedding.from_pretrained(torch.from_numpy(char_embedding_weights))
        self.char_embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                           embedding_dim=self.char_embedding_size)
        nn.init.uniform_(self.char_embedding.weight.data, -0.32, 0.32)
        # nn.init.uniform_(self.char_embedding.weight, -0.32, 0.32)
        # nn.init.uniform_(self.char_embedding.bias, 0, 0)

        #分别使用卷积窗口宽度1，3，5的卷积核进行卷积，并相加。
        self.win_sizes = [1, 3, 5]
        # self.padding = 1

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.char_embedding_size,
                      out_channels=self.char_hidden_size,
                      padding=w // 2,  # 输入的每一条边补充0的层数
                      kernel_size=w)
            for w in self.win_sizes
        ])

        self.linear = nn.Linear(len(self.win_sizes) * self.char_hidden_size, self.char_hidden_size)
        self.dropout = nn.Dropout(config.drop_rate)
        self.dropout_embed = nn.Dropout(config.drop_embed_rate)

    def conv_and_pool(self, x, conv):
        conv_out = F.relu(conv(x))
        out = F.max_pool1d(conv_out, conv_out.size(2))
        return out

    # 根据输入长度计算卷积输出长度
    # stride=1时，一维卷积层输出大小(宽度) = 序列大小 + 2*pad - 窗口大小 + 1
    # def conv_out_size(self, L):
    #     # stride = 1
    #     return (L + 2*self.padding - self.win_size) + 1 

    def forward(self, chars):  # (batch_size, max_seq_len, max_wd_len)
        batch_size, max_seq_len, max_wd_len = chars.size()

        chars = chars.reshape((-1, max_wd_len))  # (batch_size * max_seq_len, max_wd_len)

        embed_x = self.char_embedding(chars)  # (batch_size * max_seq_len, max_wd_len, char_embedding_size)  ?

        # batch_size * max_len * embedding_size ->batch_size * embedding_size * max_len
        embed_x = embed_x.permute(0, 2, 1)  # (batch_size * max_seq_len, char_embedding_size, max_wd_len)
    
        if self.training:
            embed_x = self.dropout_embed(embed_x)

        # (batch_size * max_seq_len, char_hidden_size, conv_out) ->
        # (batch_size * max_seq_len, char_hidden_size, 1)
        out = [self.conv_and_pool(embed_x, conv) for conv in self.convs]
        conv_out = torch.cat(tuple(out), dim=1)  # 对应第二个维度拼接起来，如 5*2*1,5*3*1的拼接变成5*5*1

        # 使用掩码处理padding过的不定长序列卷积结果
        # 1、根据实际序列长度计算卷积输出的长度
        # 2、在step 1的基础上生成mask
        # 3、mask乘以卷积输出的结果
        # conv_lens = self.conv_out_size(wd_lens.flatten())  # (batch_size, max_seq_len) -> (batch_size*max_seq_len, )
        # mask = torch.zeros_like(conv_out, device=chars.device)
        # for i, conv_len in enumerate(conv_lens):
        #     mask[i, :, :conv_len].fill_(1)
        # conv_out = conv_out * mask

        conv_out = conv_out.squeeze()  # 去掉维度为1的值去掉

        if self.training:
            conv_out = self.dropout(conv_out)

        out = self.linear(conv_out)

        out = out.reshape(batch_size, max_seq_len, -1)

        return out
