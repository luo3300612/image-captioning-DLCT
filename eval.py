import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from data import build_image_field
from models import model_factory

import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import multiprocessing
from evaluation import PTBTokenizer, Cider
import json

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def predict_captions(model, dataloader, text_field,cider,args):
    import itertools
    tokenizer_pool = multiprocessing.Pool()
    res = {}
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, ((detections, boxes, grids, masks), caps_gt) in enumerate(iter(dataloader)):
            detections = detections.to(device)
            boxes = boxes.to(device)
            grids = grids.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(detections, 20, text_field.vocab.stoi['<eos>'], args.beam_size, out_size=1,**{'boxes': boxes, 'grids': grids, 'masks': masks})
            caps_gen = text_field.decode(out, join_words=False)
            caps_gen1 = text_field.decode(out)
            caps_gt1 = list(itertools.chain(*([c, ] * 1 for c in caps_gt)))

            caps_gen1, caps_gt1 = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen1, caps_gt1])
            reward = cider.compute_score(caps_gt1, caps_gen1)[1].astype(np.float32)
            # reward = reward.mean().item()

            for i,(gts_i, gen_i) in enumerate(zip(caps_gt1,caps_gen1)):
                res[len(res)] = {
                    'gt':caps_gt1[gts_i],
                    'gen':caps_gen1[gen_i],
                    'cider':reward[i].item(),
                }

            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen, spice=args.spice)
    if not args.only_test:
        json.dump(res,open(args.dump_json,'w'))
    return scores


class Config:
    fusion = True
    n_hop = 4


if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--grid_on', action='store_true')
    parser.add_argument('--image_field', type=str, default="ImageDetectionsField")
    parser.add_argument('--model', type=str, default="transformer_fix")
    parser.add_argument('--max_detections', type=int, default=50)
    parser.add_argument('--grid_embed', action='store_true', default=False)
    parser.add_argument('--box_embed', action='store_true', default=False)
    parser.add_argument('--dim_feats', type=int, default=2048)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--spice', action='store_true', default=False)
    parser.add_argument('--dump_json', type=str, default='')
    parser.add_argument('--only_test',action='store_true',default=False)
    parser.add_argument('--beam_size',type=int,default=5)

    args = parser.parse_args()
    print('DLCT Evaluation')
    # Pipeline for image regions
    if args.grid_on:
        max_detections = 49
    else:
        max_detections = 50
    image_field = build_image_field(args)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    _, _, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    ref_caps_test = list(test_dataset.text)
    cider_test = Cider(PTBTokenizer.tokenize(ref_caps_test))

    # Model and dataloaders
    Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention = model_factory(args)
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention,
                                 d_in=args.dim_feats,
                                 d_k=args.d_k,
                                 d_v=args.d_v,
                                 h=args.head
                                 )
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'],
                                      d_k=args.d_k,
                                      d_v=args.d_v,
                                      h=args.head
                                      )
    
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder,args=args).to(device)

    data = torch.load(args.model_path)
    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)
    scores = predict_captions(model, dict_dataloader_test, text_field,cider_test,args)
    bleu = scores['BLEU']
    for item in bleu:
        print(item, end=' ')
    print(scores['ROUGE'], end=' ')
    print(scores['METEOR'], end=' ')
    if args.spice:
        print(scores['SPICE'], end=' ')
    print(scores['CIDEr'])
    print(scores)
