# 中文词性标注系统 (Chinese POS Tagging)

本项目为西北工业大学2021年度秋季学期自然语言处理课程大作业——中文词性标注系统的代码，作者为李奇、高云鹏。模型已部署到网站，运行app.py(或运行predict.bat)，之后打开浏览器进入http://127.0.0.1/5000即可。

### 代码依赖

本代码使用 python 3.7 开发，代码依赖 torch，numpy 和 matplotlib 运行。

可使用以下命令在conda env中安装依赖环境：

```
conda create -n ml python=3.7
pip install -r requirements.txt
```



### 代码说明

#### `Dataloader.py`

`Dataloader.py：数据处理，包括加载数据集，打包成Instance实体，捆绑成batch，生成词典等。`

#### `Vocab.py`

`Vocab.py：定义了两个类分别是词表和字符表。`

#### `congfig.py`

`config.py：模型配置文件，其中保存了用来训练模型的各个参数，以及输入模型，图像和预处理文件的地址。`

#### `rnn_easy.py`

`rnn_easy.py：LSTM模型架构。`

#### `TaggerModel_easy.py`

`TaggerModel_easy.py：搭建模型框架，实现将character-ltsm层的输出和word_embeddings层输出的词向量进行拼接等。`

#### `train_easy.py`

`train_easy.py：模型训练文件，其中含有预处理文件读入，模型训练，模型BLEU值计算，模型保存等函数。`



### 训练模型

```
当前代码中使用的batch_size为64。
# 创建文件夹
mkdir model corpus
# 下载数据（CTB中文书库或其他已经标注好的数据集）,放置在corpus中
# 准备预训练好的词向量
（下载地址：https://gitcode.net/mirrors/embedding/chinese-word-vectors?utm_source=csdn_github_accelerator）
# 开始训练（运行echo.bat）
python train_easy.py --cuda 1 --epochs 20 --hidden_size 128 --batch_size 64 --drop_rate 0.3 --drop_embed_rate 0.3 --learnging_rate 1e-4 --weight_decay 1e-6
```



### 演示Demo

演示需要训练好的翻译模型，可先运行代码进行训练：

```cmd
python predict.py 
```

执行将自动生成程序中输入的样例，样例格式如下所示：

```
Data
data_dir : corpus
train_data_path : corpus/train.tsv
test_data_path : corpus/test.tsv
dev_data_path : corpus/dev.tsv
word_embedding_path : corpus/wd_embed.txt
char_embedding_path : corpus/char_vectors.txt
Save
model_dir : model
load_vocab_path : model/vocab.pkl
save_vocab_path : model/vocab.pkl
load_char_vocab_path : model/char_vocab.pkl
save_char_vocab_path : model/char_vocab.pkl
load_model_path : model/pos_model.pkl
save_model_path : model/pos_model.pkl
Optimizer
learning_rate : 2e-3
weight_decay : 1e-7
Network
epochs : 20
nb_layers : 1
max_len : 100
hidden_size : 128
char_hidden_size : 64
batch_size : 32
drop_rate : 0.3
drop_embed_rate : 0.3
['我', '的', '名字', '叫', '张三', '！']
['PN', 'DEG', 'NN', 'VV', 'NR', 'PU']
```





