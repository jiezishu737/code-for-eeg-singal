import math
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange
#from model_3d import DenseNet_3D


#added by wang
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCEA3D(nn.Module):
    def __init__(self, channels, reduction=16, groups=4):
        super(DCEA3D, self).__init__()
        self.channels = channels

        # 动态核选择单元 - 1D版本
        self.kernel_selector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 序列维度池化
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, 3, 1),
            nn.Softmax(dim=1)
        )

        # 多尺度通道注意力分支
        self.conv3 = nn.Conv1d(16, 1, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv1d(16, 1, kernel_size=5, padding=2, bias=False)
        self.conv7 = nn.Conv1d(16, 1, kernel_size=7, padding=3, bias=False)

        # 序列位置注意力分支
        self.position_att = nn.Sequential(
            nn.Conv1d(channels, channels // groups, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // groups, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 门控融合单元
        self.gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入形状: [batch_size, seq_len, channels]
        b, s, c = x.size()

        # 转置为通道优先格式: [batch, channels, seq_len]
        x_t = x.permute(0, 2, 1)

        # ===== 动态核选择 =====
        kernel_weights = self.kernel_selector(x_t)  # [b, 3, 1]
        k3_w, k5_w, k7_w = kernel_weights[:, 0], kernel_weights[:, 1], kernel_weights[:, 2]

        # ===== 通道注意力分支 =====
        gap = torch.mean(x_t, dim=2, keepdim=True)  # [b, c, 1]
        gap_vec = gap.view(b, c, 1)  # [b, c, 1]

        # 多核并行处理
        att3 = self.conv3(gap_vec)  # [b, c, 1]
        att5 = self.conv5(gap_vec)
        att7 = self.conv7(gap_vec)

        # 加权融合
        channel_att = (k3_w.view(b, 1, 1) * att3 +
                       k5_w.view(b, 1, 1) * att5 +
                       k7_w.view(b, 1, 1) * att7)

        channel_att = self.sigmoid(channel_att)  # [b, c, 1]

        # ===== 位置注意力分支 =====
        position_att = self.position_att(x_t)  # [b, 1, s]

        # ===== 协同注意力融合 =====
        # 通道增强: [b, c, s]
        channel_enhanced = x_t * channel_att

        # 位置增强: [b, c, s]
        position_enhanced = x_t * position_att

        # 门控融合
        concat_features = torch.cat([channel_enhanced, position_enhanced], dim=1)  # [b, 2*c, s]
        fusion_gate = self.gate(concat_features)  # [b, c, s]

        output = fusion_gate * channel_enhanced + (1 - fusion_gate) * position_enhanced


        # 恢复原始形状: [batch_size, seq_len, channels]
        return output.permute(0, 2, 1),channel_att.squeeze(-1), position_att.squeeze(1)








class DCEA(nn.Module):
    def __init__(self, channels, reduction=16, groups=4):
        super(DCEA, self).__init__()
        self.channels = channels

        # 动态核选择单元
        self.kernel_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 3, 1),
            nn.Softmax(dim=1)
        )

        # 多尺度通道注意力分支
        self.conv3 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        self.conv7 = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)

        # 空间上下文分支
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // groups, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // groups, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 门控融合单元
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()

        # ===== 动态核选择 =====
        kernel_weights = self.kernel_selector(x)  # [b, 3, 1, 1]
        k3_w, k5_w, k7_w = kernel_weights[:, 0], kernel_weights[:, 1], kernel_weights[:, 2]

        # ===== 通道注意力分支 =====
        gap = F.adaptive_avg_pool2d(x, 1)  # [b, c, 1, 1]
        gap_vec = gap.view(b, c, 1)  # [b, c, 1]

        # 多核并行处理
        att3 = self.conv3(gap_vec)  # [b, c, 1]
        att5 = self.conv5(gap_vec)
        att7 = self.conv7(gap_vec)

        # 加权融合

        channel_att = (k3_w.view(b,1,1)* att3 + (k5_w.view(b,1,1) * att5) + (k7_w.view(b,1,1) * att7))
        channel_att = self.sigmoid(channel_att).view(b,c,1,1)



        # ===== 空间上下文分支 =====
        spatial_att = self.spatial_att(x)  # [b, 1, h, w]

        # ===== 协同注意力融合 =====
        channel_enhanced = x * channel_att
        spatial_enhanced = x * spatial_att

        # 门控融合
        concat_features = torch.cat([channel_enhanced, spatial_enhanced], dim=1)
        fusion_gate = self.gate(concat_features)

        output = fusion_gate * channel_enhanced + (1 - fusion_gate) * spatial_enhanced

        return output


#added by wang
class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
          
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x*y.expand_as(x)

#added by wang
class ImprovedECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ImprovedECALayer, self).__init__()
        self.channels = channels
        self.gamma = gamma
        self.b = b

        # 动态计算卷积核大小
        self.k_size = self.get_kernel_size()

        # 多尺度卷积
        self.conv1 = nn.Conv1d(1, 1, kernel_size=self.k_size, padding=(self.k_size - 1) // 2, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)

        # 自适应权重调整
        self.adaptive_weight = nn.Parameter(torch.ones(3))

        self.fusion_conv = nn.Conv1d(3,1,kernel_size=1)

        # 非线性激活
        self.nonlinear = nn.GELU()

    def get_kernel_size(self):
        k = int(abs((math.log2(self.channels) + self.b) / self.gamma))
        return k if k % 2 else k + 1

    def forward(self, x):
        #print('x.shape:', x.shape)
        # 全局平均池化
        y = F.adaptive_avg_pool2d(x,1)

        # 多尺度卷积
        y1 = self.conv1(y)
        y3 = self.conv3(y)
        y5 = self.conv5(y)

        y = self.fusion_conv(y)


        # 自适应权重融合
        y = self.adaptive_weight[0] * y1 + self.adaptive_weight[1] * y3 + self.adaptive_weight[2] * y5

        # 非线性激活
        y = self.nonlinear(y)

        # Sigmoid激活
        y = torch.sigmoid(y)

        # 通道注意力
        return x * y.expand_as(x)

#added by wang
class ImprovedECALayer_new(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ImprovedECALayer_new, self).__init__()
        self.channels = channels

        # 动态计算多尺度卷积核（确保不同）
        self.k1 = self._get_kernel_size(gamma, b)
        self.k2 = self._get_kernel_size(gamma * 0.8, b + 1)  # 差异化参数
        self.k3 = self._get_kernel_size(gamma * 1.2, b - 1)

        # 多尺度卷积（确保核为奇数且≥3）
        self.conv1 = nn.Conv1d(1, 1, kernel_size=self.k1, padding=(self.k1 - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=self.k2, padding=(self.k2 - 1) // 2, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=self.k3, padding=(self.k3 - 1) // 2, bias=False)

        # 自适应权重调整（加入温度系数)
        self.adaptive_weight = nn.Parameter(torch.ones(3))
        self.temperature = 1.0  # 可学习参数

        # 激活函数
        self.nonlinear = nn.GELU()

    def _get_kernel_size(self, gamma, b):
        k = int(abs((math.log2(self.channels) + b) / gamma))
        k = k if k % 2 else k + 1
        return max(k, 3)

    def forward(self, x):
        # 全局平均池化
        y = F.adaptive_avg_pool2d(x, 1)

        # 多尺度卷积 + 激活
        y1 = self.nonlinear(self.conv1(y))
        y2 = self.nonlinear(self.conv2(y))
        y3 = self.nonlinear(self.conv3(y))

        # 自适应融合（带温度调节）
        weights = torch.softmax(self.adaptive_weight / self.temperature, dim=0)
        y = weights[0] * y1 + weights[1] * y2 + weights[2] * y3

        # 调整维度
        #y = y.transpose(-1, -2).unsqueeze(-1)
        y = torch.sigmoid(y)

        return x * y.expand_as(x)
        


#added by wang
class reconstruction(nn.Module):
    def __init__(self, n_chan=56, fs=128, N_F=64, tem_kernelLen=0.1):
        super(reconstruction, self).__init__()
        self.n_chan = 56
        self.N_F = 56
        self.fs = 128

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_chan, (n_chan, 1), padding="same", bias=True),
            # Permute2d((2, 0, 3, 1)),
            nn.BatchNorm2d(56, eps=1e-3, momentum=0.99)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(56, N_F, (1, int(fs * tem_kernelLen)), padding="same", bias=True),
            nn.BatchNorm2d(N_F, eps=1e-3, momentum=0.99)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(N_F, N_F, (1, int(fs * tem_kernelLen)), padding="same", bias=True),
            nn.BatchNorm2d(N_F, eps=1e-3, momentum=0.99)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(N_F, n_chan, (n_chan, 1), padding="same", bias=True),
            nn.BatchNorm2d(n_chan, eps=1e-3, momentum=0.99)
        )

        self.conv5 = nn.Conv2d(n_chan, 1, (n_chan, 1), padding="same", bias=True)

    def forward(self, x):
        x = x.unsqueeze(1)
        # x = x.permute(2, 0, 3, 1)
        # encoder
        x = self.conv1(x)
        x = self.conv2(x)
        # decoder
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.conv5(x)

        x = x.squeeze(1)

        return x




#added by wang
class AFT_FULL(nn.Module):

      def __init__(self, d_model, n=64, simple=False):

          super(AFT_FULL, self).__init__()
          self.fc_q = nn.Linear(d_model, d_model)
          self.fc_k = nn.Linear(d_model, d_model)
          self.fc_v = nn.Linear(d_model, d_model)
          if(simple):
            self.position_biases = torch.zeros((n, n))
          else:
            self.position_biases = nn.Parameter(torch.ones((n, n)))
          self.d_model = d_model
          self.n = n
          self.sigmoid = nn.Sigmoid()

          self.init_weights()

      def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

      def forward(self, input):

        bs, n, dim = input.shape
        print("imput_shape is ", input.shape)

        q = self.fc_q(input) #bs,n,dim
        k = self.fc_k(input).view(1,bs,n,dim) #1,bs,n,dim
        v = self.fc_v(input).view(1,bs,n,dim) #1,bs,n,dim

        numerator = torch.sum(torch.exp(k+self.position_biases.view(n,1,-1,1))*v,dim=2) #n,bs,dim
        denominator = torch.sum(torch.exp(k+self.position_biases.view(n,1,-1,1)),dim=2) #n,bs,dim

        out=(numerator/denominator) #n,bs,dim
        out=self.sigmoid(q)*(out.permute(1,0,2)) #bs,n,dim

        return out

#added by wang new attention
class GCSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(GCSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1))
        self.qkv = nn.Conv2d(dim, dim*4, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*4, dim*4, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim*4, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        bs, len_seq, _ = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'bs (head c) len_seq len_seq -> bs head c (len_seq len_seq)', head=self.num_heads)
        k = rearrange(k, 'bs (head c) len_seq len_seq -> bs head c (len_seq len_seq)', head=self.num_heads)
        v = rearrange(v, 'bs (head c) len_seq len_seq -> bs head c (len_seq len_seq)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out,'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out













class Attention_1(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)

        self.Conv_key = nn.Conv1d(in_channels=emb_size, out_channels=emb_size*4, kernel_size=1, padding=0)
        self.Conv_query = nn.Conv1d(in_channels=emb_size, out_channels=emb_size, kernel_size=1, padding=0)
        self.Conv_value = nn.Conv1d(in_channels=emb_size, out_channels=emb_size, kernel_size=1, padding=0)



        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)


        self.pool_2q = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_2k = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.pool_1q = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.pool_1k = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.act = nn.GELU()

        self.dwconv = nn.Conv2d(32, 32, 3, padding=1, groups=1 )
        self.qkvl = nn.Conv2d(16, 64,1,padding=0)




    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        # 使用 chunk 均匀分割成两块（若长度为奇数，第一块多一个元素）
        x1, x2 = torch.chunk(x, 2, dim=1)
        len1, len2 = x1.size(1), x2.size(1)





        # 第一个注意力：query=x2, key=x1, value=x1
        k = self.key(x1).reshape(batch_size, len1, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x1).reshape(batch_size, len1, self.num_heads, -1).transpose(1, 2)
        q = self.query(x2).reshape(batch_size, len2, self.num_heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k)
        attn_12 = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn_12, v)
        out = out.transpose(1, 2)
        out_x1_x2 = out.reshape(batch_size, len2, -1)

        # 第二个注意力：自注意力 on x1
        k = self.key(x1).reshape(batch_size, len1, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x1).reshape(batch_size, len1, self.num_heads, -1).transpose(1, 2)
        q = self.query(x1).reshape(batch_size, len1, self.num_heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k)
        attn_11 = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn_11, v)
        out = out.transpose(1, 2)
        out_x1 = out.reshape(batch_size, len1, -1)

        # 第三个注意力：query=x1, key=x2, value=x2
        k = self.key(x2).reshape(batch_size, len2, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x2).reshape(batch_size, len2, self.num_heads, -1).transpose(1, 2)
        q = self.query(x1).reshape(batch_size, len1, self.num_heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k)
        attn_21 = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn_21, v)
        out = out.transpose(1, 2)
        out_x2_x1 = out.reshape(batch_size, len1, -1)

        # 第四个注意力：自注意力 on x2
        k = self.key(x2).reshape(batch_size, len2, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x2).reshape(batch_size, len2, self.num_heads, -1).transpose(1, 2)
        q = self.query(x2).reshape(batch_size, len2, self.num_heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k)
        attn_22 = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn_22, v)
        out = out.transpose(1, 2)
        out_x2 = out.reshape(batch_size, len2, -1)

        # 拼接所有输出，总长度 = len1 + len2 + len1 + len2 = 2 * seq_len
        out = torch.cat([out_x1_x2, out_x2_x1, out_x2, out_x1], dim=1)
        out = self.to_out(out)

        attn_dict = {
            'attn_11': attn_11,  # X1自注意力 [b, num_heads, S/2, S/2]
            'attn_22': attn_22,  # X2自注意力
            'attn_12': attn_12,  # X2->X1交叉
            'attn_21': attn_21  # X1->X2交叉
        }




        return out,attn_dict


class Attention_2(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # print("seq_len",seq_len,self.num_heads)
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale

        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out





#added by wang
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()

        # Spatial Feature Extraction (Channel-wise Attention)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, d_model * 4, kernel_size=(1, 1)),  # Channel expansion
            nn.BatchNorm2d(d_model * 4),
            nn.GELU(),
            nn.Conv2d(d_model * 4, d_model * 4, kernel_size=(c_in, 1)),  # Spatial aggregation
            nn.BatchNorm2d(d_model * 4),
            nn.GELU(),
            SqueezeExcitation(d_model * 4)  # Channel attention
        )

        # Temporal Feature Extraction (Depthwise Separable Conv)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(d_model * 4, d_model * 4, kernel_size=(1, 8),
                      padding='same', groups=d_model * 4),  # Depthwise
            nn.Conv2d(d_model * 4, d_model * 4, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(d_model * 4),
            nn.GELU(),
            TemporalAttention(d_model * 4)  # Temporal attention
        )

        # Feature Fusion and Projection
        self.projection = nn.Sequential(
            nn.Conv2d(d_model * 4, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )

        self.position_embedding = PositionalEmbedding(d_model)

    def forward(self, x):
        x = x.unsqueeze(1)  # [Batch, 1, C_in, T]

        # Spatial processing
        x = self.spatial_conv(x)  # [Batch, D*4, 1, T]

        # Temporal processing
        x = self.temporal_conv(x)  # [Batch, D*4, 1, T]

        # Feature fusion
        x = self.projection(x)  # [Batch, D, 1, T]
        x = x.squeeze(2).permute(0, 2, 1)  # [Batch, T, D]

        # Positional encoding
        x = x + self.position_embedding(x)
        return x


class SqueezeExcitation(nn.Module):
    """Channel attention module"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.se(x)
        return x * weights


class TemporalAttention(nn.Module):
    """Temporal attention module"""

    def __init__(self, channels):
        super().__init__()
        self.ta = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=(1, 1)),
            nn.GELU(),
            nn.Conv2d(channels // 8, 1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.ta(x.mean(dim=3, keepdim=True))  # [B,1,1,T]
        return x * attn








class Refine(nn.Module):
    def __init__(self, c_in):
        super(Refine, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.GELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class MyAttention_1(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.attention_layer = Attention_1(emb_size, num_heads, dropout=0.1)

    def forward(self, x):
        out, attn_dict = self.attention_layer(x)
        return out, attn_dict


class MyAttention_2(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.attention_layer = Attention_2(emb_size, num_heads, dropout=0.1)

    def forward(self, x):
        x = self.attention_layer(x)
        return x




class AttnRefine_1(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.attention = MyAttention_1(emb_size, num_heads)
        self.conv_layer = Refine(emb_size)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(emb_size, 4)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x_src, attn_dict = self.attention(x)
        x_src = self.conv_layer(x_src)
        gap = self.gap(x_src.permute(0, 2, 1))
        out = self.out(self.flatten(gap))
        return x_src, out, attn_dict  # 返回注意力权重

class AttnRefine_2(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.attention = MyAttention_2(emb_size, num_heads)
        self.conv_layer = Refine(emb_size)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(emb_size, 4)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x_src = self.attention(x)
        x_src = self.conv_layer(x_src)
        gap = self.gap(x_src.permute(0, 2, 1))
        out = self.out(self.flatten(gap))
        # print(out.shape)
        return x_src, out





class Eeg_csp(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        d_model = 16
        emb_size = d_model
        num_heads = 8

        self.token_embedding = TokenEmbedding(c_in=channel_size, d_model=d_model)

        self.flatten = nn.Flatten()
        self.out = nn.Linear(4, 2)
        self.stack1 = AttnRefine_1(16, 8)
        self.stack2 = AttnRefine_2(16, 8)
        
        #self.eca = ImprovedECALayer_new(16)

        self.eca = DCEA3D(16)

        self.position_embedding = PositionalEmbedding(16)
        self.reconstuction = reconstruction()
        
        

    def forward(self, x):
        #x_src = self.reconstuction(x)
        #x_src = x_src + self.position_embedding(x_src)

        x_src = self.token_embedding(x)


        new_x = []
        
        x_src_eca, channel_att1, pos_att1 = self.eca(x_src)  # 接收注意力权重
        x_src = x_src_eca + x_src
        
        x_src1, new_src1, attn_dict1 = self.stack1(x_src)  # 接收注意力权重
        
        #new_x.append(new_src1)
        
        x_src1_eca, channel_att2, pos_att2 = self.eca(x_src1)
        x_src1 = x_src1_eca + x_src1

        x_src2, new_src2 = self.stack2(x_src1)
        new_x.append(new_src2)

        out = torch.cat(new_x, -1)
        out = self.flatten(out)
        out = self.out(out)

        attn_dict = {
            'channel_att1': channel_att1,  # [b, c]
            'pos_att1': pos_att1,  # [b, s]
            'channel_att2': channel_att2,
            'pos_att2': pos_att2,
            'dcsam_attn1': attn_dict1,  # 第一个 DCSAM 的注意力矩阵
            # 注意：stack2 的注意力未被收集，如需可扩展
        }


        return out,attn_dict