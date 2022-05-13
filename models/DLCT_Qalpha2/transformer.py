import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel
from .. import position_encoding as pe


class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder, args=None):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

        self.grid_embedding = pe.PositionEmbeddingSine(256, normalize=True)
        self.box_embedding = nn.Linear(4, 512)
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_pos_embedding(self, boxes, grids,split=False):
        bs = boxes.shape[0]
        region_embed = self.box_embedding(boxes)
        grid_embed = self.grid_embedding(grids.view(bs, 7, 7, -1))
        if not self.args.box_embed:
            # print('reach here')
            region_embed = torch.zeros_like(region_embed)
        if not self.args.grid_embed:
            # print('reach here')
            grid_embed = torch.zeros_like(grid_embed)
        if not split:
            pos = torch.cat([region_embed, grid_embed], dim=1)
            return pos
        else:
            return region_embed,grid_embed

    def forward(self, regions, boxes, grids,aligns, seq, *args):
        # all_visual = torch.cat([images, grids], dim=1)
        region_embed,grid_embed = self.get_pos_embedding(boxes, grids,split=True)
        # print(all_visual.shape)
        enc_output, mask_enc = self.encoder(regions=regions, grids=grids,boxes=boxes,aligns=aligns, region_embed=region_embed,grid_embed=grid_embed)
        pos = torch.cat([region_embed, grid_embed], dim=1)
        dec_output = self.decoder(seq, enc_output, mask_enc, pos=pos)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        bs = visual.shape[0]
        boxes = kwargs['boxes']
        grids = kwargs['grids']
        aligns = kwargs['masks']
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                # all_visual = torch.cat([visual, grids], dim=1)
                self.region_embed,self.grid_embed = self.get_pos_embedding(boxes, grids,split=True)
                self.enc_output, self.mask_enc = self.encoder(regions=visual, grids=grids,boxes=boxes,aligns=aligns, region_embed=self.region_embed,grid_embed=self.grid_embed)
                self.pos = torch.cat([self.region_embed,self.grid_embed], dim=1)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc, pos=self.pos)


class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
