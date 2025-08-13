"""
文件名: model.py
Transformer Seq2Seq模型定义
用于学习从噪声命题到逆否命题的转换
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class LogicTransformer(nn.Module):
    """
    用于逻辑推理的Transformer Seq2Seq模型
    """
    
    def __init__(self, vocab_size, d_model=128, nhead=8, num_encoder_layers=3, 
                 num_decoder_layers=3, dim_feedforward=512, max_len=100):
        super(LogicTransformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer核心
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=False  # 使用 (seq_len, batch, features) 格式
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        前向传播
        
        Args:
            src: 源序列 (seq_len, batch_size)
            tgt: 目标序列 (seq_len, batch_size)
            src_mask: 源序列mask
            tgt_mask: 目标序列mask
            src_key_padding_mask: 源序列padding mask
            tgt_key_padding_mask: 目标序列padding mask
        """
        
        # 词嵌入和位置编码
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        
        src_emb = self.pos_encoding(src_emb)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Transformer前向传播
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # 输出投影
        output = self.output_projection(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """生成因果mask，防止模型看到未来的token"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        """编码器前向传播"""
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        
        memory = self.transformer.encoder(
            src_emb, 
            mask=src_mask, 
            src_key_padding_mask=src_key_padding_mask
        )
        return memory
    
    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """解码器前向传播"""
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        output = self.transformer.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        output = self.output_projection(output)
        return output


def create_padding_mask(seq, pad_token):
    """创建padding mask"""
    return (seq == pad_token)


def create_model(vocab_size, device='cpu'):
    """创建并初始化模型"""
    model = LogicTransformer(
        vocab_size=vocab_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        max_len=100
    )
    
    model = model.to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型创建成功!")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    return model


def inference(model, src_tokens, tokenizer, device='cpu', max_length=50):
    """
    推理函数：给定输入序列，生成输出序列
    
    Args:
        model: 训练好的模型
        src_tokens: 输入token序列
        tokenizer: tokenizer对象
        device: 设备
        max_length: 最大生成长度
    
    Returns:
        生成的token序列
    """
    model.eval()
    
    with torch.no_grad():
        # 准备输入
        src = torch.tensor(src_tokens).unsqueeze(1).to(device)  # (seq_len, 1)
        
        # 编码
        memory = model.encode(src)
        
        # 初始化解码序列
        tgt_tokens = [tokenizer.START_TOKEN]
        
        for _ in range(max_length):
            tgt = torch.tensor(tgt_tokens).unsqueeze(1).to(device)  # (seq_len, 1)
            
            # 创建因果mask
            tgt_mask = model.generate_square_subsequent_mask(len(tgt_tokens)).to(device)
            
            # 解码
            output = model.decode(tgt, memory, tgt_mask=tgt_mask)
            
            # 获取下一个token
            next_token_logits = output[-1, 0, :]  # 最后一个时间步的输出
            next_token = torch.argmax(next_token_logits).item()
            
            # 如果生成了结束token，停止生成
            if next_token == tokenizer.END_TOKEN:
                break
            
            tgt_tokens.append(next_token)
        
        return tgt_tokens[1:]  # 去掉START_TOKEN


if __name__ == "__main__":
    # 测试模型创建
    from logic_utils import Tokenizer
    
    tokenizer = Tokenizer()
    model = create_model(tokenizer.vocab_size)
    
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print("模型结构:")
    print(model)
