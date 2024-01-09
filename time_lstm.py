import math
import torch
import torch.nn as nn

class Senti_Attenuation_LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        # [128,20,50]
        # input_sz:输入维度-50
        # hidden_sz:隐藏维度-50
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 修改-需要增加一个exp_long_memory的权重矩阵
        self.W_e = nn.Parameter(torch.Tensor(input_size, hidden_size))      # [50,50]
        self.b_e = nn.Parameter(torch.Tensor(hidden_size))                # [50,50]

        # i_t：输入门
        self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))      # [50,50]
        self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))     # [50,50]
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))                # [1,50]

        # f_t：遗忘门
        self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # c_t：当前时刻的记忆门
        self.U_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # o_t：输出门
        self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        # 初始化权重
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, time_sen_seqs, init_states=None):

        """
        assumes x.shape represents (batch_size, sequence_size, input_size)：批处理大小、序列长度、特征长度
        输入维度为[128,20,50]， 经过转化之后，还是[128,20,50]
        time_sen_seqs:128,20,它的每个元素是每个时间点的情感衰减率，attenuation是衰减率
        """
        # [128,20,50]
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            # [128,50]
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        # 遍历计算每个序列
        for t in range(seq_sz):
            x_t = x[:, t, :]    # [128,50]
            # 获取遗忘率张量，先取然后扩展成合适的张量形状
            forget_t = time_sen_seqs[:, t]   # [128,1]
            forget_t = torch.from_numpy(forget_t).cuda()
            forget_t = forget_t.float().cuda()
            forget_t = forget_t.unsqueeze(1)
            forget_t = forget_t.expand(forget_t.shape[0], x_t.shape[-1])
            # @ 表示矩阵相乘， *表示矩阵对应元素相乘
            # [128,50] @ [50,50] = [128,50], [128,50] @ [50,50] = [128,50] ,[1,50]广播机制变成 [128,50]
            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)     # [128,50]-记忆门
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)     # [128,50]-遗忘门
            g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)        # [128,50]-当前输入
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)     # [128,50]-输出门

            # 对长期记忆需要做衰减计算
            # 创建一个与长时记忆张量c_t形状相同，值全为衰减系数的张量
            # forget_t = torch.full_like(c_t, attenuation)    # [128,50]
            c_t = c_t * forget_t        # 遗忘计算
            c_t = torch.tanh(c_t @ self.W_e + self.b_e)

            c_t = f_t * c_t + i_t * g_t     # [128,50]*[128,50],对应元素相乘，[128,50]
            h_t = o_t * torch.tanh(c_t)     # 更新输出ht, [128,50]

            hidden_seq.append(h_t.unsqueeze(0))     # 第0维增加一个维度，[1,128,50]
        # hidden_seq:20个[1,128,50]

        # reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)   # [20,128,50]
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()    # 张量维度转化[128,20,50]
        # 输出(128,20,50),(128,50),(128,50)
        return hidden_seq, (h_t, c_t)
