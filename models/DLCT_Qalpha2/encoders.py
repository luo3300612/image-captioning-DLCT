from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer_fix_all_boxatt.attention import MultiHeadAttention
from ..relative_embedding import BoxRelationalEmbedding, GridRelationalEmbedding, AllRelationalEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None,
                pos=None):
        # print('-' * 50)
        # print('layer input')
        # print(queries[11])
        q = queries + pos
        k = keys + pos
        att = self.mhatt(q, k, values, relative_geometry_weights, attention_mask, attention_weights)
        # print('mhatt outpout')
        # print(att[11])
        att = self.lnorm(queries + self.dropout(att))
        # print('norm out')
        # print(att[11])
        ff = self.pwff(att)
        # print('ff out')
        # print(ff[11])
        # print('-' * 50)
        return ff


class CrossEncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(CrossEncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None,
                pos_source=None, pos_cross=None):
        # print('-' * 50)
        # print('layer input')
        # print(queries[11])
        q = queries + pos_source
        k = keys + pos_cross
        att = self.mhatt(q, k, values, relative_geometry_weights, attention_mask, attention_weights)
        # print('mhatt outpout')
        # print(att[11])
        att = self.lnorm(queries + self.dropout(att))
        # print('norm out')
        # print(att[11])
        ff = self.pwff(att)
        # print('ff out')
        # print(ff[11])
        # print('-' * 50)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers_region = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                         identity_map_reordering=identity_map_reordering,
                                                         attention_module=attention_module,
                                                         attention_module_kwargs=attention_module_kwargs)
                                            for _ in range(N)])
        self.layers_grid = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                       identity_map_reordering=identity_map_reordering,
                                                       attention_module=attention_module,
                                                       attention_module_kwargs=attention_module_kwargs)
                                          for _ in range(N)])
        self.region2grid = nn.ModuleList([CrossEncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                            identity_map_reordering=identity_map_reordering,
                                                            attention_module=attention_module,
                                                            attention_module_kwargs=attention_module_kwargs)
                                          for _ in range(N)])
        self.grid2region = nn.ModuleList([CrossEncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                            identity_map_reordering=identity_map_reordering,
                                                            attention_module=attention_module,
                                                            attention_module_kwargs=attention_module_kwargs)
                                          for _ in range(N)])
        self.padding_idx = padding_idx

        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, regions, grids, boxes, aligns, attention_weights=None, region_embed=None, grid_embed=None):
        # input (b_s, seq_len, d_in)
        attention_mask_region = (torch.sum(regions == 0, -1) != 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        attention_mask_grid = (torch.sum(grids == 0, -1) != 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        # box embedding
        relative_geometry_embeddings = AllRelationalEmbedding(boxes)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
                                              self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        n_regions = regions.shape[1]  # 50
        n_grids = grids.shape[1]  # 49

        region2region = relative_geometry_weights[:, :, :n_regions, :n_regions]
        grid2grid = relative_geometry_weights[:, :, n_regions:, n_regions:]
        # region2grid = relative_geometry_weights[:, :, :n_regions, n_regions:]
        # grid2region = relative_geometry_weights[:, :, n_regions:, :n_regions]
        region2all = relative_geometry_weights[:,:,:n_regions,:]
        grid2all = relative_geometry_weights[:, :, n_regions:, :]

        bs = regions.shape[0]

        outs = []
        out_region = regions
        out_grid = grids
        aligns = aligns.unsqueeze(1)  # bs * 1 * n_regions * n_grids

        tmp_mask = torch.eye(n_regions, device=out_region.device).unsqueeze(0).unsqueeze(0)
        tmp_mask = tmp_mask.repeat(bs, 1, 1, 1)  # bs * 1 * n_regions * n_regions
        region_aligns = (torch.cat([tmp_mask, aligns], dim=-1) == 0) # bs * 1 * n_regions *(n_regions+n_grids)

        tmp_mask = torch.eye(n_grids, device=out_region.device).unsqueeze(0).unsqueeze(0)
        tmp_mask = tmp_mask.repeat(bs, 1, 1, 1)  # bs * 1 * n_grids * n_grids
        grid_aligns = (torch.cat([aligns.permute(0, 1, 3, 2), tmp_mask], dim=-1)==0) # bs * 1 * n_grids *(n_grids+n_regions)

        pos_cross = torch.cat([region_embed,grid_embed],dim=-2)
        for l_region, l_grid, l_r2g, l_g2r in zip(self.layers_region, self.layers_grid, self.region2grid,
                                                  self.grid2region):
            # print('encoder layer in')
            # print(out[11])
            # print('region self att')
            out_region = l_region(out_region, out_region, out_region, region2region, attention_mask_region,
                                  attention_weights, pos=region_embed)
            # print('grid self att')
            out_grid = l_grid(out_grid, out_grid, out_grid, grid2grid, attention_mask_grid, attention_weights,
                              pos=grid_embed)

            out_all = torch.cat([out_region, out_grid], dim=1)
            # print('region cross')
            out_region = l_r2g(out_region, out_all, out_all, region2all, region_aligns, attention_weights,
                               pos_source=region_embed, pos_cross=pos_cross)

            # print('grid cross')
            out_grid = l_g2r(out_grid, out_all, out_all, grid2all, grid_aligns,
                             attention_weights, pos_source=grid_embed, pos_cross=pos_cross)
            # print('encoder layer out')
            # print(out[11])
            # outs.append(out.unsqueeze(1))
        out = torch.cat([out_region, out_grid], dim=1)
        attention_mask = torch.cat([attention_mask_region, attention_mask_grid], dim=-1)
        # outs = torch.cat(outs, 1)
        # print('encoder out')
        # print(out.view(-1)[0].item())
        return out, attention_mask


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc_region = nn.Linear(d_in, self.d_model)
        self.dropout_region = nn.Dropout(p=self.dropout)
        self.layer_norm_region = nn.LayerNorm(self.d_model)

        self.fc_grid = nn.Linear(d_in, self.d_model)
        self.dropout_grid = nn.Dropout(p=self.dropout)
        self.layer_nrom_grid = nn.LayerNorm(self.d_model)

    def forward(self, regions, grids, boxes, aligns, attention_weights=None, region_embed=None, grid_embed=None):
        mask_regions = (torch.sum(regions, dim=-1) == 0).unsqueeze(-1)
        mask_grids = (torch.sum(grids, dim=-1) == 0).unsqueeze(-1)
        # print('\ninput', input.view(-1)[0].item())
        out_region = F.relu(self.fc_region(regions))
        out_region = self.dropout_region(out_region)
        out_region = self.layer_norm_region(out_region)
        out_region = out_region.masked_fill(mask_regions, 0)

        out_grid = F.relu(self.fc_grid(grids))
        out_grid = self.dropout_grid(out_grid)
        out_grid = self.layer_nrom_grid(out_grid)
        out_grid = out_grid.masked_fill(mask_grids, 0)

        # print('out4',out[11])
        return super(TransformerEncoder, self).forward(out_region, out_grid, boxes, aligns,
                                                       attention_weights=attention_weights,
                                                       region_embed=region_embed, grid_embed=grid_embed)
