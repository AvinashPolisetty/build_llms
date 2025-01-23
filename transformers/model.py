import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self,d_model:int,vocab_size:int):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

    

class PositionalEncoding(nn.Module):

    def __init__(self,d_model:int,seq_len:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(self.seq_len,self.d_model)

        #create a vector of shape (seq_len,1)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(1000.0)/d_model))

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:].requires_grad_(False))
        return self.dropout(x)


class LayerNomlization(nn.Module):

    def __init__(self,eps:float=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim =True)
        std = x.std(dim = -1,keepdim = True)
        return self.alpha * (x - mean)/(std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)

    def forward(self,x):

        return self.linear2(torch.relu(self.linear1(x)))

    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,d_model:int,h : int,dropout : float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0,"d_model is not divisble by h"

        self.d_k = d_model // h
        self.q_w = nn.Linear(self.d_model,self.d_model) #weights of queries
        self.k_w = nn.Linear(self.d_model,self.d_model) #weights of keys
        self.v_w = nn.Linear(self.d_model,self.d_model) #weights of values 

        self.o_w = nn.Linear(self.d_model,self.d_model) #weights for output's
        self.dropout = nn.Dropout(dropout)


    #this function is used to return the attention scores 
    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value) , attention_scores

    def forward(self,q,k,v,mask):
        query = self.q_w(q)  # I/p-->(batch,seq_len,d_model) O/p -->(batch,seq_len,d_model)
        keys = self.k_w(k)   # I/p-->(batch,seq_len,d_model) O/p -->(batch,seq_len,d_model)
        value = self.v_w(v)    # I/p-->(batch,seq_len,d_model) O/p -->(batch,seq_len,d_model)
        
        # (batch,seq_len,d_model)-->(batch,seq_len,h,d_k)-->(batch,h,seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        keys = keys.view(keys.shape[0],keys.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores = MultiHeadAttentionBlock.attention(query,keys,value,mask,self.dropout)

        # (batch,h,seq_len,d_k)-->(batch,seq_len,h,d_k)-->(batch,seq_len,d_model)
        x = x.transpose(1,2).continguous().view(x.shape[0],-1,self.h * self.d_k)
        
        # I/p-->(batch,seq_len,d_model) O/p -->(batch,seq_len,d_model)
        return self.o_w(x)


class ResidualConnection(nn.Module):
    
    def __init__(self,dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNomlization()

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,feed_forward:FeedForwardBlock,dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x,src_mask):

        x = self.residual_connection[0](x,lambda x:self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connection[1(x,self.feed_forward)]

        return x


class Encoder(nn.Module):
    
    def __init__(self,layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNomlization()

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self,self_attention:MultiHeadAttentionBlock,cross_attention:MultiHeadAttentionBlock,feed_forward:FeedForwardBlock,dropout:float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    
    def forward(self,x,encoder,src_mask,tgt_mask):
        x = self.residual_connection[0](x,lambda x:self.self_attention(x,x,x,tgt_mask))
        x = self.residual_connection[1](x,lambda x:self.cross_attention(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connection[2](x,self.feed_forward)


class Decoder(nn.Module):

    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNomlization()

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):

        return torch.log_softmax(self.proj(x),dim = -1)



class Transformer(nn.Module):

    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbeddings,tgt_embed:InputEmbeddings,src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,projection:ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection

    def encoder(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)

    def decoder(self,encoder_output,src_mask,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

    def project(self,x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer