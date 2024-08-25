# src: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/multi_headed_attn.py
""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn
from c2nl.utils.misc import generate_relative_positions_matrix, \
    relative_matmul

import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, d_k, d_v, dropout=0.1,
                 max_relative_positions=0, use_neg_dist=True, coverage=False):
        super(MultiHeadedAttention, self).__init__()

        self.head_count = head_count
        self.model_dim = model_dim
        self.d_k = d_k
        self.d_v = d_v

        self.key = nn.Linear(model_dim, head_count * self.d_k)
        self.query = nn.Linear(model_dim, head_count * self.d_k)
        self.value = nn.Linear(model_dim, head_count * self.d_v)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.head_count * d_v, model_dim)
        self._coverage = coverage

        self.max_relative_positions = max_relative_positions
        self.use_neg_dist = use_neg_dist

        # for draw graph
        self.attn = None

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1 \
                if self.use_neg_dist else max_relative_positions + 1
            self.relative_positions_embeddings_k = nn.Embedding(
                vocab_size, self.d_k)
            self.relative_positions_embeddings_v = nn.Embedding(
                vocab_size, self.d_v)

    def forward(self, key, value, query, mask=None, layer_cache=None,
                attn_type=None, step=None, coverage=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):
           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        use_gpu = key.is_cuda

        def shape(x, dim):
            """  projection """
            return x.view(batch_size, -1, head_count, dim).transpose(1, 2)

        def unshape(x, dim):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                # 1) Project key, value, and query.
                key = shape(self.key(key), self.d_k)
                value = shape(self.value(value), self.d_v)
                query = shape(self.query(query), self.d_k)

                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value

            elif attn_type == "context":
                query = shape(self.query(query), self.d_k)
                if layer_cache["memory_keys"] is None:
                    key = shape(self.key(key), self.d_k)
                    value = shape(self.value(value), self.d_v)
                else:
                    key, value = layer_cache["memory_keys"], \
                                 layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = shape(self.key(key), self.d_k)
            value = shape(self.value(value), self.d_v)
            query = shape(self.query(query), self.d_k)

        if self.max_relative_positions > 0 and attn_type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions, self.use_neg_dist,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings_k(
                relative_positions_matrix.to(key.device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings_v(
                relative_positions_matrix.to(key.device))

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(self.d_k)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        if self.max_relative_positions > 0 and attn_type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        # ----------------------------
        # We adopt coverage attn described in Paulus et al., 2018
        # REF: https://arxiv.org/abs/1705.04304
        exp_score = None
        if self._coverage and attn_type == 'context':
            # batch x num_heads x query_len x 1
            maxes = torch.max(scores, 3, keepdim=True)[0]
            # batch x num_heads x query_len x key_len
            exp_score = torch.exp(scores - maxes)

            if step is not None:  # indicates inference mode (one-step at a time)
                if coverage is None:
                    # t = 1 in Eq(3) from Paulus et al., 2018
                    unnormalized_score = exp_score
                else:
                    # t = otherwise in Eq(3) from Paulus et al., 2018
                    assert coverage.dim() == 4  # B x num_heads x 1 x key_len
                    unnormalized_score = exp_score.div(coverage + 1e-20)
            else:
                multiplier = torch.tril(torch.ones(query_len - 1, query_len - 1))
                # batch x num_heads x query_len-1 x query_len-1
                multiplier = multiplier.unsqueeze(0).unsqueeze(0). \
                    expand(batch_size, head_count, *multiplier.size())
                multiplier = multiplier.cuda() if scores.is_cuda else multiplier

                # B x num_heads x query_len-1 x key_len
                penalty = torch.matmul(multiplier, exp_score[:, :, :-1, :])
                # B x num_heads x key_len
                no_penalty = torch.ones_like(penalty[:, :, -1, :])
                # B x num_heads x query_len x key_len
                penalty = torch.cat([no_penalty.unsqueeze(2), penalty], dim=2)
                assert exp_score.size() == penalty.size()
                unnormalized_score = exp_score.div(penalty + 1e-20)

            # Eq.(4) from Paulus et al., 2018
            attn = unnormalized_score.div(unnormalized_score.sum(3, keepdim=True))

        # Softmax to normalize attention weights
        else:
            # 3) Apply attention dropout and compute context vectors.
            attn = self.softmax(scores).to(query.dtype)

        # ----------------------------

        # 3) Apply attention dropout and compute context vectors.
        # attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        if self.max_relative_positions > 0 and attn_type == "self":
            context = unshape(context_original
                              + relative_matmul(drop_attn,
                                                relations_values,
                                                False),
                              self.d_v)
        else:
            context = unshape(context_original, self.d_v)

        final_output = self.output(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # a list of size num_heads containing tensors
        # of shape `batch x query_len x key_len`
        attn_per_head = [attn.squeeze(1)
                         for attn in attn.chunk(head_count, dim=1)]
        
        #### for draw graph ###
        #self.attn=attn_per_head

        covrage_vector = None
        if (self._coverage and attn_type == 'context') and step is not None:
            covrage_vector = exp_score  # B x num_heads x 1 x key_len

        return final_output, attn_per_head, covrage_vector

    def update_dropout(self, dropout):
        self.dropout.p = dropout

class SingleNodeAttentionLayer(nn.Module):
    
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(SingleNodeAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W1 = nn.Parameter(torch.empty(size=(in_features, 2*out_features))) #nn.Linear(in_features, 2*out_features)
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.a1 = nn.Parameter(torch.empty(size=(2*out_features, 1))) #nn.Linear(2*out_features, 1)
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)

        self.W2 = nn.Parameter(torch.empty(size=(in_features, 2*out_features))) #nn.Linear(in_features, 2*out_features)
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.a2 = nn.Parameter(torch.empty(size=(2*out_features, 1))) #nn.Linear(2*out_features, 1)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.W3 = nn.Parameter(torch.empty(size=(in_features, 2*out_features))) #nn.Linear(in_features, 2*out_features)
        nn.init.xavier_uniform_(self.W3.data, gain=1.414)
        self.a3 = nn.Parameter(torch.empty(size=(2*out_features, 1))) #nn.Linear(2*out_features, 1)
        nn.init.xavier_uniform_(self.a3.data, gain=1.414)

        self.W4 = nn.Parameter(torch.empty(size=(in_features, 2*out_features))) #nn.Linear(in_features, 2*out_features)
        nn.init.xavier_uniform_(self.W4.data, gain=1.414)
        self.a4 = nn.Parameter(torch.empty(size=(2*out_features, 1))) #nn.Linear(2*out_features, 1)
        nn.init.xavier_uniform_(self.a4.data, gain=1.414)

        self.Wo = nn.Parameter(torch.empty(size=(8*out_features, out_features))) #nn.Linear(8*out_features, out_features)
        nn.init.xavier_uniform_(self.Wo.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, features): # [b, max_node, max_token, token_emsize]

        W1_features = torch.matmul(features, self.W1) # [b, max_node, max_token, 2 * out_feature]
        simple_attention1 = F.softmax(torch.matmul(W1_features,self.a1),dim=2) # [b, max_node, max_token, 1]
        weighted_node_feature1 = torch.matmul(simple_attention1.permute(0,1,3,2), W1_features).squeeze(2) # [b, max_node, 2 * out_feature]
        
        W2_features = torch.matmul(features, self.W2)
        simple_attention2 = F.softmax(torch.matmul(W2_features,self.a2),dim=2)
        weighted_node_feature2 = torch.matmul(simple_attention2.permute(0,1,3,2), W2_features).squeeze(2)

        W3_features = torch.matmul(features, self.W3)
        simple_attention3 = F.softmax(torch.matmul(W3_features,self.a3),dim=2)
        weighted_node_feature3 = torch.matmul(simple_attention3.permute(0,1,3,2), W3_features).squeeze(2)

        W4_features = torch.matmul(features, self.W4)
        simple_attention4 = F.softmax(torch.matmul(W4_features,self.a4),dim=2)
        weighted_node_feature4 = torch.matmul(simple_attention4.permute(0,1,3,2), W4_features).squeeze(2)

        weighted_node_feature_multi = torch.cat([weighted_node_feature1, weighted_node_feature2, weighted_node_feature3, weighted_node_feature4], dim=2) # [b, max_node, 8 * out_feature]
        #print("weighted_node_feature_multi",weighted_node_feature_multi.shape)
        
        #print("h",h.shape)
        #print("self.Wo",self.Wo.shape)
        h = F.elu(torch.matmul(weighted_node_feature_multi, self.Wo)) # [b, max_node, out_feature]
        
        #print("nodes", len(features))
        #print("h",h.shape)
        #print("all_token_tensor", all_token_tensor.shape)

        #exit()
        #with open('../../data/test/look_npy.txt', 'a', encoding='utf-8') as outfile:
        #    outfile.write(str(self.a1.data.detach().cpu().numpy()) + '\n')
        return h  # 返回所有结点的单一向量组成的二维张量特征数据
    
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True, training=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.training = training

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(3*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adjacency, node2node_features): # [b, max_node, in_feature], [b, max_node, max_node], [b, max_node * max_node, token_emsize]
        
        # 以下部分参数不一样要分别计算
        h = F.dropout(h, self.dropout, training=self.training)
        
        Wh = torch.matmul(h, self.W) # [b, max_node, out_feature]
        
        a_input = self._prepare_attentional_mechanism_input(Wh, node2node_features) # (b, max_node, max_node, 3 * out_features)
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) # (b, max_node, max_node)

        zero_vec = -9e15*torch.ones_like(e) # (b, max_node, max_node)

        attention = torch.where(adjacency > 0, e, zero_vec) # (b, max_node, max_node)
        
        attention = F.softmax(attention, dim=2)  #exp（-无穷）= 0， 0/正数 = 0  即对非边的注意力设置为0，只关注相邻边
        
        attention = F.dropout(attention, self.dropout, training=self.training)
        #print("self.training",self.training)
        if self.concat:
            #edge_attr = F.elu(torch.mm(edge_attr, self.W)) # [num_edge, out_feature]
            h_prime = F.elu(torch.matmul(attention, Wh)) # [b, max_node, out_feature]
        else:
            #edge_attr = torch.mm(edge_attr, self.W)
            h_prime = torch.matmul(attention, Wh)

        return h_prime #, edge_attr

    def _prepare_attentional_mechanism_input(self, Wh, node2node_features): # [b, max_node, out_feature], [b, max_node * max_node, token_emsize]
        N = Wh.size()[1] # number of nodes
        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN,            e1, e2, ..., eN, ...,            e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)   # [b, max_node * max_node, out_feature]
        Wh_repeated_alternating = Wh.repeat(1, N, 1) # [b, max_node * max_node, out_feature]
        node2node_features = torch.matmul(node2node_features, self.W) # [b, max_node * max_node, out_feature]

        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1  || 1->1
        # e1 || e2  || 1->2
        # e1 || e3  || 1->3
        # ...
        # e1 || eN  || 1->N
        # e2 || e1  || 2->1
        # e2 || e2  || 2->2
        # e2 || e3  || 2->3
        # ...
        # e2 || eN  || 2->N
        # ...
        # eN || e1  || N->1
        # eN || e2  || N->2
        # eN || e3  || N->3
        # ...
        # eN || eN  || N->N

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating, node2node_features], dim=2) # (b, max_node * max_node, 3 * out_features)

        return all_combinations_matrix.view(Wh.size(0), N, N, 3 * self.out_features) # (b, max_node, max_node, 3 * out_features)
