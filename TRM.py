import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import torch.utils.data as Data
import random
import modbus_tk
import modbus_tk.modbus_tcp as modbus_tcp
from numpy import *
import check_tcp as ch
LOGGER = modbus_tk.utils.create_logger("console")
import rouletee_select as rou_select
import self_adaption as adaption
torch.manual_seed(3407)
num_data = 0
mal_num  = 0
#Mun_arr表示表示变异概率矩阵，总共8个，从0-7
nor_factor = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]  #[0.2] * 8
abn_factor = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]  #[0.2] * 8
error_factor = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]   #[0.2] * 8

def  build_key():
    key = {}
    str1 = ['a', 'b', 'c', 'd', 'e', 'f', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    num = {'SS': 256, 'EE': 257, 'PP': 258,"":259}
    n = 0
    for x in str1:
        for y in str1:
           keys = x+y
           key[keys] = n
           n= n + 1
    key.update(num)
    return  key

src_vocab = build_key()
src_vocab_size = len(src_vocab)
src_idx2word = {i: w for i, w in enumerate(src_vocab)}

tgt_vocab = build_key()
tgt_vocab_size = len(tgt_vocab)
idx2word = {i: w for i, w in enumerate(tgt_vocab)}

device = 'cpu'
def make_batch(str1):
    input_batch = [[src_vocab[n] for n in str1.split()]]
    return input_batch
def make_data(sentences):
    """把单词序列转换为数字序列"""
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    # print(sentences)
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        dec_input = [[src_vocab[n] for n in sentences[i][1].split()]]  # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_output = [[src_vocab[n] for n in sentences[i][2].split()]]  # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

## 10
def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]

## 7. ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        ## 输入进来的维度分别是 [batch_size x n_heads x len_q x d_k]  K： [batch_size x n_heads x len_k x d_k]  V: [batch_size x n_heads x len_k x d_v]
        ##首先经过matmul函数得到的scores形状是 : [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        ## 然后关键词地方来了，下面这个就是用到了我们之前重点讲的attn_mask，把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

## 6. MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        ## 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):

        ## 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;
        ##输入进来的数据形状： Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        ##下面这个就是先映射，后分头；一定要注意的是q和k分头之后维度是一致额，所以一看这里都是dk
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        ## 输入进行的attn_mask形状是 batch_size x len_q x len_k，然后经过下面这个代码得到 新的attn_mask : [batch_size x n_heads x len_q x len_k]，就是把pad信息重复了n个头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)


        ##然后我们计算 ScaledDotProductAttention 这个函数，去7.看一下
        ## 得到的结果有两个：context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

## 8. PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class MyDataSet(Data.Dataset):
    """自定义DataLoader"""

    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

## 3. PositionalEncoding 代码实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)## 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)##这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置
        ## 上面代码获取之后得到的pe:[max_len*d_model]

        ## 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  ## 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

## 5. EncoderLayer ：包含两个部分，多头注意力机制和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()  #多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()    #前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):
        ## 下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model] 需要注意的是最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数 6.
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

## 2. Encoder 部分包含三个部分：词向量embedding，位置编码部分，注意力层及后续的前馈神经网络
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)  ## 这个其实就是去定义生成一个矩阵，大小是 src_vocab_size * d_model
        self.pos_emb = PositionalEncoding(d_model) ## 位置编码情况，这里是固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_En_layers)]) ## 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；
    def forward(self, enc_inputs):
        ## 这里我们的 enc_inputs 形状是： [batch_size x source_len]

        ## 下面这个代码通过src_emb，进行索引定位，enc_outputs输出形状是[batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)

        ## 这里就是位置编码，把两者相加放入到了这个函数里面，从这里可以去看一下位置编码函数的实现；3.
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        ##get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响，去看一下这个函数 4.
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            ## 去看EncoderLayer 层函数 5.
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

## 10.
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

## 9. Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_de_layers)])
    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, tgt_len, d_model]

        ## get_attn_pad_mask 自注意力层的时候的pad 部分
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)

        ## get_attn_subsequent_mask 这个做的是自注意层的mask部分，就是当前单词之后看不到，使用一个上三角为1的矩阵
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        ## 两个矩阵相加，大于0的为1，不大于0的为0，为1的在之后就会被fill到无限小
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        ## 这个做的是交互注意力机制中的mask矩阵，enc的输入是k，我去看这个k里面哪些是pad符号，给到后面的模型；注意哦，我q肯定也是有pad符号，但是这里我不在意的，之前说了好多次了哈
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns
## 1. 从整体网路结构来看，分为三个部分：编码层，解码层，输出层
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()  ## 编码层
        self.decoder = Decoder()  ## 解码层
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False) ## 输出层 d_model 是我们解码层每个token输出的维度大小，之后会做一个 tgt_vocab_size 大小的softmax

    def forward(self, enc_inputs, dec_inputs):
        # 这里有两个数据进行输入，一个是enc_inputs 形状为[batch_size, src_len]，主要是作为编码段的输入，一个dec_inputs，形状为[batch_size, tgt_len]，主要是作为解码端的输入

        # enc_inputs作为输入 形状为[batch_size, src_len]，输出由自己的函数内部指定，想要什么指定输出什么，可以是全部tokens的输出，可以是特定每一层的输出；也可以是中间某些参数的输出；
        # enc_outputs就是主要的输出，enc_self_attns这里没记错的是QK转置相乘之后softmax之后的矩阵值，代表的是每个单词和其他单词相关性；
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # dec_outputs 是decoder主要输出，用于后续的linear映射； dec_self_attns类比于enc_self_attns 是查看每个单词对decoder中输入的其余单词的相关性；dec_enc_attns是decoder中每个单词对encoder中每个单词的相关性；
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # dec_outputs做映射到词表大小
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
def softmax1(data):
    data = list(data)
    max_tensor = torch.tensor(data)
    _, index1 = max_tensor.topk(10)
    index2 = sorted(list(index1.squeeze().numpy()))
    index2.append(-1)
    denominator = 0
    j = 1
    l = len(data)
    for i in data:
        denominator = denominator + math.exp(i / j)
    for i in range(0, l):
        data[i] = math.exp(data[i] / j) / denominator
    return data
def range_bit_decoder(model,memory_arr,Mun_id1,Mun_arr1,rouletee_arr1,enc_input,start_symbol):
    """变异采样编码
    """
    #fc 表示功能码
    #sign_num表示标记矩阵

    sign_pro = [0] * 40   #记录位置的变异矩阵，当生成一条测试用例开始时，置0
    sign_hex =[]   #标记变异的字节是什么
    hex_list = [i for i in range(256)] # 代表16进制的数组
    other_str = [257, 259]
    for i in other_str:
        hex_list.append(i)
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    Mut_output = []
    Nor_output = []
    number_id = 0

    while not terminal:
        # 预测阶段：dec_input序列会一点点变长（每次添加一个新预测出来的单词）
        dec_input = torch.cat([dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)], -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        soft_max=softmax1(projected.squeeze(0).data.numpy()[len(projected.squeeze(0).data)-1])
        max_data=max(soft_max)
        prob = soft_max.index(max_data)
        ture_output = prob
        Nor_output.append(ture_output)

        if torch.tensor(prob) == tgt_vocab["PP"]:
            Mut_output.append(tgt_vocab["PP"])
        elif torch.tensor(prob) == tgt_vocab["EE"]:
            Mut_output.append(tgt_vocab["EE"])
        else:
            fuzz = random.uniform(0, 1)
            if fuzz < Mun_arr1[Mun_id1][number_id]:
                # 发生变异
                sign_pro[number_id] = 1
                ture_output = rou_select.roulette_selection(hex_list, rouletee_arr1[Mun_id1])

                # ture_output = rou_select.roulette_selection1(hex_list, rouletee_arr1[Mun_id1], memory_arr,Mun_id1)
                # ture_output = random.randint(0,255)
                sign_hex.append(ture_output)
            Mut_output.append(ture_output)

        next_symbol = torch.tensor(prob)
        number_id = number_id + 1
        if next_symbol == tgt_vocab["EE"] or torch.tensor(prob) == tgt_vocab["EE"]:
            terminal = True
    return torch.tensor(Mut_output), torch.tensor(Nor_output), sign_pro, sign_hex

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
def data_change(str1):
    data=""
    for i in range(0, len(str1), 2):
        data=data+str1[i] + str1[i + 1]+" "
    return data.strip()
def add_data(str1,max_len):
    data=""
    for i in range(0, len(str1), 2):
        data = data + str1[i] + str1[i + 1] + " "
    for i in range(0,int((max_len-len(str1))/2)):
        data = data+"PP "
    return data.strip()
def data_variation(str2):
    return str2
#####################
#概率矩阵调整
def change_abn_arr(change_arr1,sign_arr1,number_abnormal,Mun_id):#异常情况下，概率矩阵调整
    recode_number = 0
    global abn_factor
    for arr_1 in sign_arr1:
        if arr_1  == 1:
            change_arr1[recode_number] = change_arr1[recode_number]+(1-change_arr1[recode_number])*abn_factor[Mun_id]
            # change_arr1[recode_number] = change_arr1[recode_number] + (1 - change_arr1[recode_number]) * 0.2
            # change_arr1[recode_number] = change_arr1[recode_number] + (1 - change_arr1[recode_number]) * 0.002
        recode_number = recode_number + 1
    abn_factor[Mun_id] = adaption.adaption_factor(1,number_abnormal)
    #print("异常",Mun_id, abn_factor)
    return change_arr1

def change_bn_arr(change_arr2,sign_arr2,number_normal,Mun_id):#正常情况下，概率矩阵调整
    recode_number = 0
    global nor_factor
    for arr_1 in sign_arr2:
        if arr_1  == 1:
            change_arr2[recode_number] = change_arr2[recode_number] + (1 - change_arr2[recode_number]) * nor_factor[Mun_id]
            # change_arr2[recode_number] = change_arr2[recode_number] + (1 - change_arr2[recode_number]) * 0.2
            # change_arr2[recode_number] = change_arr2[recode_number] + (1 - change_arr2[recode_number]) * 0.002
        recode_number = recode_number + 1
    nor_factor[Mun_id] = adaption.adaption_factor(0, number_normal)
    #print("正常",Mun_id, nor_factor)
    return change_arr2

def change_err_arr(change_arr3,sign_arr3,number_error,Mun_id):#错误情况下，概率矩阵调整
    recode_number = 0
    global error_factor
    for arr_1 in sign_arr3:
        if arr_1  == 1:
            change_arr3[recode_number] = change_arr3[recode_number]-change_arr3[recode_number]*error_factor[Mun_id]
            # change_arr3[recode_number] = change_arr3[recode_number] - change_arr3[recode_number] * 0.2
            # change_arr3[recode_number] = change_arr3[recode_number] - change_arr3[recode_number] * 0.002
        recode_number = recode_number + 1
    error_factor[Mun_id] = adaption.adaption_factor(-1,number_error)
    #print("错误",Mun_id,error_factor)
    return change_arr3
#########################################################################
def change_rou_arr(change_arr4,sign_num1): #更改轮盘赌的结果 和 适应度值
    for x in sign_num1:
        if x==257:
            x=256
        elif x == 259:
            x=257
        change_arr4[x] = change_arr4[x] + 2
    # print("增加的字节",change_arr4)
    return change_arr4

def change_rou_arr1(sign_num1,id,memory_arr,fc): #更改轮盘赌的结果 和 适应度值
    for x in sign_num1:
        if x==257:
            x=256
        elif x == 259:
            x=257
        memory_arr[x][id] = memory_arr[x][id] + 1
        memory_arr[x][0] = fc
    # print("增加的字节",change_arr4)
    return memory_arr
##############################################################
##模型初次训练
def train_text():
    # optimizer1 = optim.Adam(model.parameters(), lr=0.005)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    num = 0
    ftxt = open("modbus_tcp.txt", "r")
    sentences = []
    fd = ftxt.readlines()
    max_len1 = len(fd[fd.index(max(fd, key=len))]) - 18
    random.shuffle(fd)

    for str1 in fd:
        input = []
        data = str1.split()
        input.append(data_change(data[0]))
        input.append("SS " + add_data(data[1], max_len1))
        input.append(add_data(data[1], max_len1) + " EE")
        sentences.append(input)
    ftxt.close()
 #
    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 32, True)
    for epoch in range(10):
        num = num + 1
        s_time = time.time()
        l = []
        for enc_inputs, dec_inputs, dec_outputs in loader:
            optimizer.zero_grad()
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            l.append(loss.data.item())
            loss.backward()
            optimizer.step()
        e_time = time.time()
        epoch_mins, epoch_secs = epoch_time(s_time, e_time)
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}:'.format(sum(l) / len(l)),
              f'Time: {epoch_mins}m {epoch_secs}s')
        torch.save(model, 'New_Model/Modbus_1E1D' + str(num) + '.pth')
#测试用例生成######################
def gen_text(Model_name):
    print("调用的模型：",Model_name)
    b_time = time.time()
    try:

        init_val = [-1, -1, -1]
        memory_arr = [init_val[:] for _ in range(258)]
        # 连接从机地址,这里要注意端口号和IP与从机一致
        MASTER = modbus_tcp.TcpMaster(host="127.0.0.1", port=502)
        MASTER.set_timeout(0.8)
        LOGGER.info("connected")
        model = torch.load(Model_name)

        ftxt1 = open("modbus_tcp.txt", "r")
        fd1 = ftxt1.readlines()
        number = 0
#######变异初始数据
        unusual_data = 0
        num_data = 0
        num0 = 0
#######未变异初始数据
        Nor_unusual_data = 0
        Nor_num_data = 0
        Nor_num0 = 0
        #8个概率矩阵，对应8种功能码
        Mun_arr = [[] for _ in range(8)]
        for i in range(8):
            Mun_arr[i] = [0.5] * 40
        #8个轮盘赌矩阵，对应每个功能码的选择
        rouletee_arr = [[] for _ in range(8)]
        for i in range(8):
            rouletee_arr[i] = [1] * 258

        record_all_text_1 = [[] for _ in range(8)]
        record_right_text_1 = [[] for _ in range(8)]
        #
        record_all_text = []
        record_right_text = []

        abn_number = [0] * 8
        nor_number = [0] * 8
        error_number = [0] * 8
        #ftxt_recode = open('C:\\Users\\10165\Desktop\\new_result\\ModRSsim2_1.txt', 'w')
        # record_0f=open('0F1.txt', 'w+')
        # ftxt_recode = open("ModRSsim2_mutation.txt", 'w')
        receive_arr = []
        ab_arr = []
        while number <= 16001:
            time.sleep(0.2)
            uid = random.randint(0, 3999)
            length = random.randint(0, 3999)
            slave_fc = random.randint(0, 3999)
            str_uid = fd1[uid][0:4]
            str_len = fd1[length][8:12]
            str_slave = fd1[slave_fc][12:16]
            fc = fd1[slave_fc][14:16]
            str0 = str_uid + "0000" + str_len + str_slave

            enc_inputs = make_batch(data_change(str0))
            number = number + 1

            Mun_dict = {  # 获取功能码对应的变异矩阵
                "01": 0,
                "02": 1,
                "03": 2,
                "04": 3,
                "05": 4,
                "06": 5,
                "10": 6,
                "0f": 7,
            }
            Mun_id = Mun_dict.get(fc, 7)
            Mun_b,Nor_b,sign_arr,sign_hex = range_bit_decoder(model,memory_arr,Mun_id,Mun_arr,rouletee_arr,torch.tensor(enc_inputs).view(1, -1).to(device),start_symbol=tgt_vocab["SS"])

############经过变异的测试
            input_txt = " ".join('%s' % idx2word[id] for id in torch.tensor(enc_inputs).numpy()[0]).replace(" ", "")
            data1 = ''.join([idx2word[n.item()] for n in Mun_b.squeeze()]).replace("PP", "").replace("SS", "")
            ##############
            record_all_text.append(fc+data1.replace("EE", ""))
            record_all_text_1[Mun_id].append(data1.replace("EE", ""))
            #记录所有测试用例中不重复的个数
            ##############
            modbus_tcp1 = input_txt + "" + data1.replace("EE", "")
            lenth = int(len(modbus_tcp1[12:]) / 2)
            length = hex(lenth).split("0x")
            length1 = length[1]
            for i in range(0, 4 - len(length[1])):
                length1 = "0" + length1
            string = list(modbus_tcp1)
            string[8:12] = length1
            modbus_tcp1 = "".join(string)   ####变异后的输出

            flag, reasion = ch.check(modbus_tcp1)
            num_data = num_data + 1
            num,fc_code = MASTER.execute1(modbus_tcp1[12:14], modbus_tcp1)

            if flag == False and num == 0:
                record_right_text.append(fc+data1.replace("EE", ""))
                record_right_text_1[Mun_id].append(data1.replace("EE", ""))
                num0 = num0 + 1
                unusual_data = unusual_data + 1
                abn_number[Mun_id] = abn_number[Mun_id] + 1
                # 概率调节
                Mun_arr[Mun_id] = change_abn_arr(Mun_arr[Mun_id],sign_arr,abn_number[Mun_id],Mun_id)
                # 轮盘赌
                rouletee_arr[Mun_id] = change_rou_arr(rouletee_arr[Mun_id],sign_hex)
                # memory_arr = change_rou_arr1(sign_hex,1, memory_arr,Mun_id)
                # ftxt_recode.write(modbus_tcp1+"\n")
                # ftxt_recode.write(fc_code+"\n")
                #调节概率矩阵，异常的情况下，调整概率矩阵

            elif num == 0:
                record_right_text.append(fc+data1.replace("EE", ""))
                record_right_text_1[Mun_id].append(data1.replace("EE", ""))
                #记录测试用例中不重复的个数
                num0 = num0 + 1
                nor_number[Mun_id] = nor_number[Mun_id] + 1

                Mun_arr[Mun_id] = change_bn_arr(Mun_arr[Mun_id],sign_arr,nor_number[Mun_id],Mun_id)
                # memory_arr = change_rou_arr1(sign_hex,2, memory_arr,Mun_id)

                # rouletee_arr[Mun_id] = change_rou_arr(rouletee_arr[Mun_id], sign_hex)
                # 调节概率矩阵，正常的情况下，调整概率矩阵
            else:
                error_number[Mun_id] = error_number[Mun_id] + 1
                Mun_arr[Mun_id] = change_err_arr(Mun_arr[Mun_id],sign_arr,error_number[Mun_id],Mun_id)
                # memory_arr = change_rou_arr1(sign_hex,3, memory_arr,Mun_id)

                # 调节概率矩阵，错误的情况下，调整概率矩阵
            target_numbers = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]

            o_time = time.time()
            if num_data in target_numbers:
                new_record_text = list(set(record_all_text))
                new_right_text = list(set(record_right_text))
                record_len = len(new_record_text)
                right_len = len(new_right_text)
                epoch_mins, epoch_secs = epoch_time(b_time, o_time)
                print("###########################第",num_data,"测试用例################################")
                print("测试", num_data, "条数据", f'共用时: {epoch_mins}m {epoch_secs}s')
                print("      接收率：",format(num0 / num_data, '.4%'))
                receive_arr.append(format(num0 / num_data, '.4%'))
                print("      错误率：", format(1 - (num0 / num_data), '.4%'))
                print("非正常正确数据：",unusual_data,format((unusual_data / num_data), '.4%'))
                ab_arr.append(format(unusual_data / num_data, '.4%'))
                print("所有测试用例中不重复的测试用例个数：", record_len, format(record_len / num_data, '.4%'))
                print("正确测试用例中不重复的测试用例个数：", right_len, format(right_len / num0, '.4%'))
                # ftxt_recode.write("共测试" + str(num_data) + "条数据," + str(f'共用时: {epoch_mins}m {epoch_secs}s') + "\n")
                # ftxt_recode.write("      接收率：" + str(num0) + "  " + str(format(num0 / num_data, '.4%')) + "\n")
                # ftxt_recode.write("      错误率：" + str(format(1 - (num0 / num_data), '.4%')) + "\n")
                # ftxt_recode.write("非正常正确数据：" + str(unusual_data) + "  " + str(format((unusual_data / num_data), '.4%')) + "\n")
                # ftxt_recode.write("所有测试用例中不重复的测试用例个数：" + str(record_len) + "  " + str(format(record_len / num_data, '.4%')) + "\n")
                # ftxt_recode.write("正确测试用例中不重复的测试用例个数：" + str(right_len) + "  " + str(format(right_len / num0, '.4%')) + "\n")
        # ftxt_recode.write("总体接受率："+ str(receive_arr))
        # ftxt_recode.write("总体异常率：" + str(ab_arr))
        print("总体接受率：", receive_arr)
        print("总体异常率：", ab_arr)
        # ftxt_recode.close()

    except modbus_tk.modbus.ModbusError as err:
        LOGGER.error("%s- Code=%d" % (err, err.get_exception_code()))


##########################
if __name__ == '__main__':
    ## 模型参数
    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_En_layers = 1
    n_de_layers = 1
    # n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    model = Transformer()
    criterion = nn.CrossEntropyLoss()
    chose_num = int(input("1.最初模型训练; 2.测试用例生成"))
    if chose_num == 1: # 模型生成
        train_text()
    elif chose_num == 2: #测试用例生成
        name11 = [
                  # "Modbus_Model/Modbus_1E1D37.pth"
                  "New_Model/Modbus_1E1D"
                 ]
        for gen in range(10,11):
            # print("####################",Model_name,"####################")
           # Model_name = name11[0]+str(gen)+'.pth'
            Model_name = "Modbus_Model/Modbus_1E1D37.pth"
            gen_text(Model_name)  # (num1,Mod_num) num1是训练的代数  Mod_num代的第n次模型保存


