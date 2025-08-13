"""
文件名: model.py
Transformer Seq2Seq模型定义
用于学习从噪声命题到逆否命题的转换
支持CUDA加速和混合精度训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


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


def create_cuda_model(vocab_size: int, device: str = 'auto',
                     use_mixed_precision: bool = True, **kwargs) -> Tuple[LogicTransformer, torch.device]:
    """
    创建CUDA优化的模型

    Args:
        vocab_size: 词汇表大小
        device: 设备选择 ('auto', 'cpu', 'cuda', 'cuda:0'等)
        use_mixed_precision: 是否使用混合精度
        **kwargs: 模型参数

    Returns:
        (model, device): 模型和设备
    """
    try:
        from cuda_utils import CUDAManager

        # 自动选择最佳设备
        if device == 'auto':
            cuda_manager = CUDAManager()
            device = cuda_manager.device
            cuda_manager.optimize_cuda_settings()
        else:
            device = torch.device(device)

        # 创建模型
        model = LogicTransformer(
            vocab_size=vocab_size,
            d_model=kwargs.get('d_model', 128),
            nhead=kwargs.get('nhead', 8),
            num_encoder_layers=kwargs.get('num_encoder_layers', 3),
            num_decoder_layers=kwargs.get('num_decoder_layers', 3),
            dim_feedforward=kwargs.get('dim_feedforward', 512),
            max_len=kwargs.get('max_len', 100)
        )

        # 移动到指定设备
        model = model.to(device)

        # 混合精度优化
        if use_mixed_precision and device.type == 'cuda':
            # 检查是否支持混合精度
            props = torch.cuda.get_device_properties(device)
            if props.major >= 7:  # Volta架构及以上
                # 将模型转换为半精度（在需要时）
                # 注意：实际的混合精度训练通过GradScaler实现
                logger.info("✅ 模型支持混合精度训练")
            else:
                logger.warning(f"⚠️ GPU计算能力{props.major}.{props.minor}不支持高效混合精度")
                use_mixed_precision = False

        # 编译模型（PyTorch 2.0+）
        if hasattr(torch, 'compile') and device.type == 'cuda':
            try:
                model = torch.compile(model, mode='default')
                logger.info("🚀 模型编译优化已启用")
            except Exception as e:
                logger.warning(f"模型编译失败: {e}")

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"🚀 CUDA模型创建成功!")
        logger.info(f"📍 设备: {device}")
        logger.info(f"📊 总参数: {total_params:,}")
        logger.info(f"🎯 可训练参数: {trainable_params:,}")
        logger.info(f"🔥 混合精度: {'启用' if use_mixed_precision else '禁用'}")

        # 估算模型内存使用
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
        logger.info(f"💾 估算模型大小: {model_size_mb:.1f}MB")

        return model, device

    except ImportError:
        logger.warning("CUDA工具不可用，回退到CPU模式")
        device = torch.device('cpu')
        model = LogicTransformer(vocab_size=vocab_size, **kwargs)
        return model, device


def optimize_model_for_inference(model: LogicTransformer, device: torch.device) -> LogicTransformer:
    """
    为推理优化模型

    Args:
        model: 训练好的模型
        device: 目标设备

    Returns:
        优化后的模型
    """
    model.eval()

    # 如果是CUDA设备，进行额外优化
    if device.type == 'cuda':
        # 启用cudnn基准模式
        torch.backends.cudnn.benchmark = True

        # 尝试使用TorchScript优化
        try:
            # 创建示例输入
            vocab_size = model.vocab_size
            sample_src = torch.randint(0, vocab_size, (10, 1), device=device)
            sample_tgt = torch.randint(0, vocab_size, (10, 1), device=device)

            # 转换为TorchScript
            traced_model = torch.jit.trace(model, (sample_src, sample_tgt))
            traced_model = torch.jit.optimize_for_inference(traced_model)

            logger.info("✅ TorchScript优化完成")
            return traced_model

        except Exception as e:
            logger.warning(f"TorchScript优化失败: {e}")

    return model


def get_model_memory_usage(model: LogicTransformer, device: torch.device) -> Dict[str, float]:
    """
    获取模型内存使用情况

    Args:
        model: 模型
        device: 设备

    Returns:
        内存使用信息字典
    """
    if device.type != 'cuda':
        return {'error': 'Only available for CUDA devices'}

    # 计算模型参数内存
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

    # 计算缓冲区内存
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())

    # 获取GPU内存信息
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)

    return {
        'param_memory_mb': param_memory / (1024 * 1024),
        'buffer_memory_mb': buffer_memory / (1024 * 1024),
        'total_model_memory_mb': (param_memory + buffer_memory) / (1024 * 1024),
        'gpu_allocated_mb': allocated / (1024 * 1024),
        'gpu_reserved_mb': reserved / (1024 * 1024)
    }


# 保持向后兼容
def create_model(vocab_size: int, **kwargs) -> LogicTransformer:
    """创建标准模型（向后兼容）"""
    return LogicTransformer(vocab_size=vocab_size, **kwargs)


if __name__ == "__main__":
    # 测试CUDA模型创建
    from logic_utils import Tokenizer

    print("🧪 测试CUDA模型创建")
    print("=" * 50)

    tokenizer = Tokenizer()

    # 测试CUDA模型
    try:
        model, device = create_cuda_model(
            vocab_size=tokenizer.vocab_size,
            device='auto',
            d_model=128,
            nhead=8
        )

        print(f"\n📊 模型信息:")
        print(f"设备: {device}")
        print(f"词汇表大小: {tokenizer.vocab_size}")

        # 获取内存使用情况
        if device.type == 'cuda':
            memory_info = get_model_memory_usage(model, device)
            print(f"模型内存: {memory_info['total_model_memory_mb']:.1f}MB")
            print(f"GPU已分配: {memory_info['gpu_allocated_mb']:.1f}MB")

        # 测试前向传播
        print("\n🔍 测试前向传播...")
        batch_size = 2
        seq_len = 10

        src = torch.randint(0, tokenizer.vocab_size, (seq_len, batch_size), device=device)
        tgt = torch.randint(0, tokenizer.vocab_size, (seq_len, batch_size), device=device)

        with torch.no_grad():
            output = model(src, tgt[:-1])
            print(f"输出形状: {output.shape}")
            print("✅ 前向传播测试成功")

    except Exception as e:
        print(f"❌ CUDA模型测试失败: {e}")

        # 回退到CPU模型
        print("🔄 回退到CPU模型...")
        model = create_model(tokenizer.vocab_size)
        print("✅ CPU模型创建成功")
