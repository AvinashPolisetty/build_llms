import torch
import torch.nn as nn
from dataclasses import dataclass
import math


class InputEmbeddings(nn.Module):

    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embeddings(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self,d_model:int,seq_len:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.arange(seq_len,d_model)

        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        x = self.dropout(x)
        return x


class LayerNormalization(nn.Module):

    def __init__(self,eps:float = 10 **-6):
        super().__init__()

        self.eps =eps
        self.beta = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self,x):

        mean = x.mean(dim = -1,keepdims = True)
        std = x.std(dim=-1,keepdims = True)
        return self.alpha * (x - mean)/(math.sqrt(std) + self.eps) + self.beta


class FeedForwardLayer(nn.Module):

    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.ln1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.Linear(d_ff,d_model)


    def forward(self,x):

        """"
        Inputs to NN : (batch,seq_len,d_model)
        hidden layer : (batch,seq_len,d_ff)
        output of NN : (batch,seq_len,d_model)
        
        """

        return self.ln2(self.dropout(torch.relu(self.ln1(x))))


class MultiHeadAttention(nn.Module):

    def __init__(self,d_model:int,h:int,dropout:float):
        super().__init__()

        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        assert d_model % h == 0 ,"d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attentionscores(query,key,value,mask,dropout:nn.Dropout):

        d_k = query.shape[-1] 

        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores = attention_scores.softmax(dim = -1) #(batch,h,seq_len,d_k)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value)   , attention_scores



    def forward(self,q,k,v,mask):

        query = self.w_q(q) #(batch,seq_len,d_model)
        key = self.w_k(k)  #(batch,seq_len,d_model)
        values = self.w_v(v)  #(batch,seq_len,d_model)

        """
        Here we have to change the dimension,so that the each head will see the (seq_len,d_k)
        """

        #(batch,seq_len,d_model)-->(batch,seq_len,h,d_k) after transpose it will become -->(batch,h,seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        values = values.view(values.shape[0],values.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores = MultiHeadAttention.attentionscores(query,key,values,mask,self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self,dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

 
class EncoderBlock(nn.Module):

    def __init__(self,self_attention:MultiHeadAttention,ffwd:FeedForwardLayer,dropout:float):
        super().__init__()
        self.self_attention = self_attention
        self.ffwd = ffwd
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        x = self.residual_connections[0](x,lambda x : self.self_attention(x,x,x,src_mask))
        x = self.residual_connections[1](x,self.ffwd)
        return x 


class Encoder(nn.Module):

    def __init__(self,layer:nn.ModuleList):
        super().__init__()
        self.layer = layer
        self.norm = LayerNormalization()

    def forward(self,x,mask):
        for layer in range(self.layer):
            x = layer(x,mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self,self_attn:MultiHeadAttention,cross_attn:MultiHeadAttention,ffwd:FeedForwardLayer,dropout:float):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ffwd = ffwd
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x = self.residual_connection[0](x,lambda x : self.self_attn(x,x,x,tgt_mask))
        x = self.residual_connection[1](x,lambda x : self.cross_attn(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connection[2](x,self.ffwd)
        return x 


class Decoder(nn.Module):

    def __init__(self,layer:nn.ModuleList):
        super().__init__()

        self.layer = layer
        self.norm = LayerNormalization()

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in range(self.layer):
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self,d_model:int,vocab_size:int,dropout:float):
        super().__init__()

        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)


class Transformer(nn.Module):

    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbeddings,tgt_embed:InputEmbeddings,src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,
                    projection:ProjectionLayer):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = projection

    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encode(src,src_mask)

    def decode(self,encoder_output,src_mask,tgt,tgt_mask):

        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decode(tgt,encoder_output,src_mask,tgt_mask)

    def projection(self,x):
        return self.proj(x)

    
def build_transformer(src_vocab:int,tgt_vocab:int,src_seqlen:int,tgt_seqlen:int,d_model:int=512,N:int =6,h:int=8,dropout:float=0.1,d_ff:int=2048):

    src_embed = InputEmbeddings(d_model,src_vocab)
    tgt_embed = InputEmbeddings(d_model,tgt_vocab)

    src_pos = PositionalEncoding(d_model,src_seqlen,dropout)
    tgt_pos = PositionalEncoding(d_model,src_seqlen,dropout)

    encoder_blocks = []

    for _ in range(N):
        encoder_selfattn = MultiHeadAttention(d_model,h,dropout)
        ffwd = FeedForwardLayer(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(d_model, encoder_selfattn, ffwd, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardLayer(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(d_model,nn.ModuleList(encoder_block))
    decoder = Decoder(d_model,nn.ModuleList(decoder_blocks))

    proj_layer = ProjectionLayer(d_model,tgt_vocab)

    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,proj_layer)


    for p in transformer.parameters():
            if p.dim > 1:
                nn.init.xavier_uniform_(p)
    
    return transformer
