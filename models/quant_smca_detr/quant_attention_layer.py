import torch
from torch.nn.functional import linear, pad
from torch import Tensor
from torch.nn import MultiheadAttention

from typing import Optional, Tuple, List
import warnings
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .lsq_plus import *
from ._quan_base_plus import *
from torch._jit_internal import boolean_dispatch, List, Optional, _overload, Tuple
from torch.overrides import has_torch_function, handle_torch_function

class DistributionAlignment(nn.Module):
    def __init__(self, head_dim):
        super(DistributionAlignment, self).__init__()
        self.head_dim = head_dim
        self.bias = nn.Parameter(torch.zeros(self.head_dim), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(1, 1, 1, self.head_dim), requires_grad=True)
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.training and self.init_state == 0:
            self.bias.data.copy_(torch.zeros(self.head_dim))
            self.scale.data.copy_(torch.ones(1, 1, 1, self.head_dim))
            self.init_state.fill_(1)
        out = (x - x.mean(-1,keepdim=True))/torch.sqrt(x.var(dim=-1, keepdim=True) + 1e-5)
        out = self.scale.expand_as(out) * out + self.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(out)
        return out

def multi_head_attention_forward(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 q_act : nn.Module,
                                 k_act : nn.Module,
                                 v_act : nn.Module,
                                 attn_act : nn.Module,
                                 norm_q: nn.Module,
                                 norm_k: nn.Module,
                                 norm_v: nn.Module,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj: nn.Module,
                                 #in_proj_bias: Tensor,
                                 bias_k: Optional[Tensor],
                                 bias_v: Optional[Tensor],
                                 add_zero_attn: bool,
                                 dropout_p: float,
                                 out_proj: nn.Module,
                                 #out_proj_bias: Tensor,
                                 training: bool = True,
                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = True,
                                 attn_mask: Optional[Tensor] = None,
                                 use_separate_proj_weight: bool = False,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 static_k: Optional[Tensor] = None,
                                 static_v: Optional[Tensor] = None
                                ) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    if not torch.jit.is_scripting():
        tens_ops = (query, key, value, in_proj.weight, in_proj.bias, bias_k, bias_v,
                    out_proj.weight, out_proj.bias)
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                multi_head_attention_forward, tens_ops, query, key, value,
                embed_dim_to_check, num_heads, in_proj.weight, in_proj.bias,
                bias_k, bias_v, add_zero_attn, dropout_p, out_proj.weight,
                out_proj.bias, training=training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)

    # print(query.shape)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    # print(use_separate_proj_weight)
    if not use_separate_proj_weight:
        # print(3333333)
        # This is inline in_proj function with in_proj_weight and in_proj_bias
        in_proj_weight = in_proj.qw(in_proj.weight)
        in_proj_bias = in_proj.bias

        # print(in_proj.weight.shape)

        # print(in_proj_weight.shape)
        # print(in_proj_bias.shape)


        # print(tgt_len)
        _b = in_proj_bias
        _start = 0
        _end = embed_dim
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = F.linear(query, _w, _b)

        # print(q.shape)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        #print(key.shape)
        _b = in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = F.linear(key, _w, _b)
        #print(_w)
        #print(_b)
        #print(k.shape)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = embed_dim * 2
        _end = None
        _w = in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        v = F.linear(value, _w, _b)

    else:
        print(2222222)
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)

    # q = q * scaling
    # print(q.shape)


    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
    if k is not None:
        k = k.contiguous().view(-1, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
    if v is not None:
        v = v.contiguous().view(-1, bsz, num_heads, head_dim).permute(1, 2, 0, 3)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(-2)

    #print(k.shape, key_padding_mask.shape)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    # print(q.shape)
    # q = q_act(q)
    if norm_q is not None:
        q = norm_q(q) #self.norm_q.normalized_shape, self.norm_k.weight, self.norm_k.bias, self.norm_k.eps)

    if norm_k is not None:
        k = norm_k(k)

    q = q_act(q)
    k = k_act(k)
    # 这里就是得到的attention
    attn_output_weights = q @ k.transpose(-2, -1) * scaling
    # Tensor:{2, 8, 462, 462}
    #torch.bmm(q, k.transpose(-1, -2))

    #print(q.shape, attn_output_weights.shape)
    assert list(attn_output_weights.size()) == [bsz, num_heads, tgt_len, src_len]

    # atten_mask = None
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask


    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    #print(attn_output_weights.shape)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    #print(attn_output_weights.shape, v.shape)
    # 进行量化
    attn_output_weights = attn_act(attn_output_weights)
    if norm_v is not None:
        v = norm_v(v)
    v = v_act(v)
    #print(attn_output_weights.shape, v.shape)

    #print(v)
    # 与v相乘得到最终的attention （462，2，256）
    attn_output = attn_output_weights @ v #.transpose(-2, -1)
    #torch.bmm(attn_output_weights, v)
    attn_output = attn_output.view(bsz * num_heads, tgt_len, head_dim)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    out_proj_weight = out_proj.qw(out_proj.weight)
    out_proj_bias = out_proj.bias
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class QuantMultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, n_bit=4, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, encoder=True):
        super(QuantMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.encoder = encoder

        if self.encoder:
            self.norm_q = DistributionAlignment(embed_dim//num_heads)
            self.norm_k = None#nn.LayerNorm(embed_dim//num_heads)
            self.norm_v = None#nn.LayerNorm(embed_dim//num_heads)
        else:
            self.norm_q = None
            self.norm_k = None
            self.norm_v = None

        self.q_act = ActLSQ(nbits_a=8, in_features=num_heads)
        self.k_act = ActLSQ(nbits_a=8, in_features=num_heads)
        self.v_act = ActLSQ(nbits_a=8, in_features=num_heads)
        self.attn_act = ActLSQ(nbits_a=8, in_features=num_heads)


        self.in_proj = LinearLSQ(embed_dim, embed_dim * 3, nbits_w = n_bit, bias=True)

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = self.in_proj.weight
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        #print(self.in_proj_weight)

        if bias:
            self.in_proj_bias = self.in_proj.bias
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = LinearLSQ(embed_dim, embed_dim, nbits_w = n_bit, bias=True)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            torch.nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            torch.nn.init.xavier_uniform_(self.q_proj_weight)
            torch.nn.init.xavier_uniform_(self.k_proj_weight)
            torch.nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            torch.nn.init.constant_(self.in_proj_bias, 0.)
            torch.nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            torch.nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            torch.nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(QuantMultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        #print('self._qkv_same_embed_dim', self._qkv_same_embed_dim)
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value,
                self.q_act, self.k_act, self.v_act, self.attn_act,
                self.norm_q, self.norm_k, self.norm_v,
                self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value,
                self.q_act, self.k_act, self.v_act, self.attn_act,
                self.norm_q, self.norm_k, self.norm_v,
                self.embed_dim, self.num_heads,
                self.in_proj,
                #self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)