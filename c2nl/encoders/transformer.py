"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np

from c2nl.modules.util_class import LayerNorm
from c2nl.modules.multi_head_attn import MultiHeadedAttention
from c2nl.modules.position_ffn import PositionwiseFeedForward
from c2nl.encoders.encoder import EncoderBase
from c2nl.utils.misc import sequence_mask

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self,
                 d_model,
                 heads,
                 d_ff,
                 d_k,
                 d_v,
                 dropout,
                 max_relative_positions=0,
                 use_neg_dist=True,
                 sytax_aware=True):
        super(TransformerEncoderLayer, self).__init__()
        
        self.attention = MultiHeadedAttention(heads,
                                              d_model,
                                              d_k,
                                              d_v,
                                              dropout=dropout,
                                              max_relative_positions=max_relative_positions,
                                              use_neg_dist=use_neg_dist)
        
        #for code structure
        if sytax_aware == True:
            self.sytax_aware = True
            self.feed_forward_mid = PositionwiseFeedForward(d_model, d_ff, dropout) 
            self.attention_struc = MultiHeadedAttention(heads,
                                                        d_model,
                                                        d_k,
                                                        d_v,
                                                        dropout=dropout,
                                                        max_relative_positions=max_relative_positions,
                                                        use_neg_dist=use_neg_dist)
        else:
            self.sytax_aware = False
            
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, mask, adjacency=None):
        """
        Transformer Encoder Layer definition.
        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`
            position_rep: for code structure `Tensor [bs, code_len, d_pos]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        
        if self.sytax_aware == True:

            '''for ablation study
            context, attn_per_head, _ = self.attention(inputs, inputs, inputs,
                                                   mask=adjacency, attn_type="self")
            hidden_state = self.layer_norm(self.dropout(context) + inputs)
            out = self.feed_forward_mid(hidden_state)
            '''
            #base
            context, attn_per_head, _ = self.attention(inputs, inputs, inputs,
                                                   mask=mask, attn_type="self")
            hidden_state = self.layer_norm(self.dropout(context) + inputs)
            mid = self.feed_forward_mid(hidden_state)

            context, attn_per_head, _ = self.attention_struc(mid, mid, mid,
                                                   mask=adjacency, attn_type="self")
            hidden_state_2 = self.layer_norm(self.dropout(context) + mid)
            out = self.feed_forward(hidden_state_2)
            
            return out, attn_per_head
        else:            
            hidden_state = inputs
            context, attn_per_head, _ = self.attention(hidden_state, hidden_state, hidden_state,
                                                       mask=mask, attn_type="self")
            out = self.layer_norm(self.dropout(context) + hidden_state) # [batch_size, max_node, heads * out_feature ( = emsize)] 
        
            return self.feed_forward(out), attn_per_head       

class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".
    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    Returns:
        (`FloatTensor`, `FloatTensor`):
        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self,
                 num_layers,
                 d_model=512,
                 heads=8,
                 d_k=64,
                 d_v=64,
                 d_ff=2048,
                 dropout=0.2,
                 max_relative_positions=0,
                 use_neg_dist=True):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        if isinstance(max_relative_positions, int):
            max_relative_positions = [max_relative_positions] * self.num_layers
        assert len(max_relative_positions) == self.num_layers

        # for code structure
        assert num_layers % 2 == 0
        self.v_layer = nn.ModuleList(
            [TransformerEncoderLayer(d_model,
                                     heads,
                                     d_ff,
                                     d_k,
                                     d_v,
                                     dropout,
                                     max_relative_positions=max_relative_positions[i],
                                     use_neg_dist=use_neg_dist,
                                     sytax_aware=True) 
             for i in range(num_layers//3)])
        self.s_layer = nn.ModuleList(
            [TransformerEncoderLayer(d_model,
                                     heads,
                                     d_ff,
                                     d_k,
                                     d_v,
                                     dropout,
                                     max_relative_positions=max_relative_positions[i],
                                     use_neg_dist=use_neg_dist,
                                     sytax_aware=False)
             for i in range(num_layers//3)])  

    def count_parameters(self):
        params = list(self.v_layer.parameters()) + list(self.s_layer.parameters()) 
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(self, src, lengths=None, adjacency_stmt=None, adjacency_graph=None):
        """
        Args:
            src (`FloatTensor`): `[batch_size x src_len x model_dim]`
            lengths (`LongTensor`): length of each sequence `[batch]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        self._check_args(src, lengths)

        out = src
        mask = None if lengths is None else \
            ~sequence_mask(lengths, out.shape[1]).unsqueeze(1)
        # for code structure
        if adjacency_stmt is not None:
            adjacency_stmt = ~(adjacency_stmt[:, :out.shape[1], :out.shape[1]].bool()) 
            adjacency_code =  ~(adjacency_graph[:, :out.shape[1], :out.shape[1]].bool()) # [batch_size, src_len, src_len] 
        # Run the forward pass of every layer of the tranformer.
        representations = []
        attention_scores = []

        for i in range(self.num_layers//3):
            out, attn_per_head = self.v_layer[i](out, adjacency_stmt, adjacency=adjacency_code) 
            representations.append(out)
            attention_scores.append(attn_per_head)
            # for code structure
            out_, attn_per_head = self.s_layer[i](out, mask) 
            out = out + out_
            representations.append(out)
            attention_scores.append(attn_per_head)
       
        return representations, attention_scores 
