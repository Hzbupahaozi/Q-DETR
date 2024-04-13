"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .attention_layer import GaussianMultiheadAttention
from .quant_attention_layer import *
from .lsq_plus import *
from ._quan_base_plus import *


class Transformer(nn.Module):
    # 多了参数n_bit = 4, dynamic_scale = 'type1', smooth = 8

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, n_bit=3,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, smooth=8, dynamic_scale=True):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, n_bit, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layers = []
        for layer_index in range(num_decoder_layers):
            decoder_layer = TransformerDecoderLayer(dynamic_scale, smooth, layer_index,
                                                    d_model, nhead, n_bit, dim_feedforward, dropout,
                                                    activation, normalize_before)
            decoder_layers.append(decoder_layer)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        if dynamic_scale in ["type2", "type3", "type4"]:
            for layer_index in range(num_decoder_layers):
                nn.init.zeros_(self.decoder.layers[layer_index].point3.weight)
                with torch.no_grad():
                    nn.init.ones_(self.decoder.layers[layer_index].point3.bias)

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    """
    传进来的四个参数：
    src={Tensor:[2,256,20,31]}
    mask{Tensor:[2,20,31]}
    query_embed:{Parameter:[100,512]}
    pos_embed:{Tensor:(2,256,20,31)}
    """

    def forward(self, src, mask, query_embed, pos_embed, h_w, reshape_outputs1, reshape_outputs2):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        # debug 这里传进来的reshape_outputs1和2会影响outputs
        # reshape_outputs1 = reshape_outputs2 =torch.zeros(300,1)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).float().to(src.device)
        grid = grid.reshape(-1, 2).unsqueeze(1).repeat(1, bs * 8, 1)

        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        query_embed, tgt = torch.split(query_embed, self.d_model, dim=1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = tgt.unsqueeze(1).repeat(1, bs, 1)  # queries
        mask = mask.flatten(1)

        # quantnoise会影响backbone的输出特征图还有就是
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # print(reshape_outputs1, reshape_outputs2)
        hs, points, D = self.decoder(grid, h_w, tgt, memory, reshape_outputs1, reshape_outputs2,
                                     memory_key_padding_mask=mask,
                                     pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), points.transpose(0, 1), D


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            # layers是多个模型的堆叠
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)
            # D = self.norm(D)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList(decoder_layer)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, grid, h_w, tgt, memory,
                reshape_outputs1, reshape_outputs2,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        points = []
        point_sigmoid_ref = None
        #
        for layer in self.layers:
            output, point, point_sigmoid_ref, D = layer(
                grid, h_w, output, memory,
                reshape_outputs1, reshape_outputs2,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos, query_pos=query_pos, point_ref_previous=point_sigmoid_ref
            )
            points.append(point)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        # True
        if self.return_intermediate:
            return torch.stack(intermediate), points[0], D

        return output.unsqueeze(0), D


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, n_bit=4, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = QuantMultiheadAttention(d_model, nhead, n_bit=n_bit, dropout=dropout, encoder=False)
        # Implementation of Feedforward model
        print(d_model)
        self.linear1 = LinearLSQ(d_model, dim_feedforward, nbits_w=n_bit)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = LinearLSQ(dim_feedforward, d_model, nbits_w=n_bit)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # 用的是这个src就是最后的输出
    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # # 计算D
        # D = self.norm1(src2)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, dynamic_scale, smooth, layer_index,
                 d_model, nhead, n_bit=4, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = QuantMultiheadAttention(d_model, nhead, n_bit=n_bit, dropout=dropout, encoder=True)
        self.multihead_attn = GaussianMultiheadAttention(d_model, nhead, n_bit=n_bit, dropout=dropout, encoder=True)
        # Implementation of Feedforward model
        self.linear1 = LinearLSQ(d_model, dim_feedforward, nbits_w=n_bit)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = LinearLSQ(dim_feedforward, d_model, nbits_w=n_bit)

        self.smooth = smooth
        self.dynamic_scale = dynamic_scale

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if layer_index == 0:
            self.point1 = MLP(256, 256, 2, 3)
            self.point2 = nn.Linear(d_model, 2 * 8)
        else:
            self.point2 = nn.Linear(d_model, 2 * 8)
        self.layer_index = layer_index
        if self.dynamic_scale == "type2":
            self.point3 = nn.Linear(d_model, 8)
        elif self.dynamic_scale == "type3":
            self.point3 = nn.Linear(d_model, 16)
        elif self.dynamic_scale == "type4":
            self.point3 = nn.Linear(d_model, 24)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, grid, h_w, tgt, memory,
                     reshape_outputs1, reshape_outputs2,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     point_ref_previous: Optional[Tensor] = None):
        tgt_len = tgt.shape[0]

        out = self.norm4(tgt + query_pos)
        point_sigmoid_offset = self.point2(out)

        q = k = self.with_pos_embed(tgt, query_pos)
        # 这里就是self_atten里面的linear的量化，如果使用quantnoise会有影响
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        # tgt2
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.layer_index == 0:
            point_sigmoid_ref_inter = self.point1(out)
            point_sigmoid_ref = point_sigmoid_ref_inter.sigmoid()
            point_sigmoid_ref = (h_w - 0) * point_sigmoid_ref / 32
            point_sigmoid_ref = point_sigmoid_ref.repeat(1, 1, 8)
        else:
            point_sigmoid_ref = point_ref_previous

        point = point_sigmoid_ref + point_sigmoid_offset
        point = point.view(tgt_len, -1, 2)
        distance = (point.unsqueeze(1) - grid.unsqueeze(0)).pow(2)

        if self.dynamic_scale == "type1":
            scale = 1
            distance = distance.sum(-1) * scale
        elif self.dynamic_scale == "type2":
            scale = self.point3(out)
            scale = scale * scale
            scale = scale.reshape(tgt_len, -1).unsqueeze(1)
            distance = distance.sum(-1) * scale
        elif self.dynamic_scale == "type3":
            scale = self.point3(out)
            scale = scale * scale
            scale = scale.reshape(tgt_len, -1, 2).unsqueeze(1)
            distance = (distance * scale).sum(-1)
        elif self.dynamic_scale == "type4":
            scale = self.point3(out)
            scale = scale * scale
            scale = scale.reshape(tgt_len, -1, 3).unsqueeze(1)
            distance = torch.cat([distance, torch.prod(distance, dim=-1, keepdim=True)], dim=-1)
            distance = (distance * scale).sum(-1)

        gaussian = -(distance - 0).abs() / self.smooth

        # 修改的地方
        # 因为batchsize是2所以就是根据一张一张图片来修改
        # 这里的query_pos.clone()真的就是很关键，不然就是query_pos的值会被置零
        reshape_object_query = (query_pos.clone().permute(1, 2, 0)).permute(0, 2, 1)
        object_query1 = reshape_object_query[0]
        object_query2 = reshape_object_query[1]
        mask1 = torch.any(reshape_outputs1 == 0, dim=1)
        object_query1[mask1] = 0
        mask2 = torch.any(reshape_outputs2 == 0, dim=1)
        object_query2[mask2] = 0
        new_object_query = torch.stack([object_query1, object_query2]).permute(1, 0, 2)
        if new_object_query == None:
            print('object为None')
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   # tgt(上一层的输出):(300,2,256) query_pos:(object query):(300,2,256)
                                   key=self.with_pos_embed(memory, pos),  # memory(660,2,256) pos:(660,2,256)
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   gaussian=[gaussian])[0]

        tgt3 = self.multihead_attn(query=self.with_pos_embed(tgt, new_object_query),
                                   # tgt(上一层的输出):(300,2,256) query_pos:(object query):(300,2,256)
                                   key=self.with_pos_embed(memory, pos),  # memory(660,2,256) pos:(660,2,256)
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   gaussian=[gaussian])[0]

        D = tgt3

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if self.layer_index == 0:
            return tgt, point_sigmoid_ref_inter, point_sigmoid_ref, D
        else:
            return tgt, None, point_sigmoid_ref, D

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, grid, h_w, tgt, memory,
                reshape_outputs1, reshape_outputs2,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                point_ref_previous: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(grid, h_w, tgt, memory,
                                 reshape_outputs1, reshape_outputs2,
                                 tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,
                                 point_ref_previous)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        n_bit=args.n_bit,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        # 多了这两个参数
        smooth=args.smooth,
        dynamic_scale=args.dynamic_scale,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
