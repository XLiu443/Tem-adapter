import torch
import numpy as np
from tqdm import tqdm
import argparse
import os, sys
import json
import pickle
from termcolor import colored
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from DataLoader import VideoQADataLoader
from utils import todevice

import model.TempAligner as TempAligner

from config import cfg, cfg_from_file
import clip
from SemanticAligner import SemanticAligner

def validate(cfg, tempaligner, semanticaligner, clip_model, data, device, write_preds=False):
    tempaligner.eval()
    clip_model.eval()
    semanticaligner.eval()
    print('validating...')
    total_acc, count = 0.0, 0
    all_preds = []
    gts = []
    v_ids = []
    q_ids = []
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            _, answers, ans_candidates, batch_clips_data, question = [todevice(x, device) for x in batch]
            if cfg.train.batch_size == 1:
                answers = answers.to(device)
            else:
                answers = answers.to(device).squeeze()
            batch_size = answers.size(0)
            feat_dim = batch_clips_data.shape[-1]
            num_ans = ans_candidates.shape[1]
            ans_candidates = ans_candidates.view(-1, 77)
            with torch.no_grad():
                answers_features = clip_model.encode_text( ans_candidates ).float()
            answers_features = semanticaligner(answers_features, batch_clips_data).float()
            question_features = clip_model.encode_text( question.squeeze() ).float()
            video_appearance_feat = batch_clips_data.view(batch_size, -1, feat_dim)
            answers_features = answers_features.view(batch_size, num_ans, -1)

            answers = answers.cuda().squeeze()
            batch_inputs = [answers,  answers_features, video_appearance_feat, question_features]
            logits, visual_embedding_decoder = tempaligner(*batch_inputs)

            logits = logits.to(device)
            preds = torch.argmax(logits.view(batch_size, 4), dim=1)
            agreeings = (preds == answers)
            if write_preds:
                if cfg.dataset.question_type not in ['action', 'transition', 'count']:
                    preds = logits.argmax(1)
                if cfg.dataset.question_type in ['action', 'transition']:
                    answer_vocab = data.vocab['question_answer_idx_to_token']
                else:
                    answer_vocab = data.vocab['answer_idx_to_token']
                for predict in preds:
                    if cfg.dataset.question_type in ['count', 'transition', 'action']:
                        all_preds.append(predict.item())
                    else:
                        all_preds.append(answer_vocab[predict.item()])
                for gt in answers:
                    if cfg.dataset.question_type in ['count', 'transition', 'action']:
                        gts.append(gt.item())
                    else:
                        gts.append(answer_vocab[gt.item()])
                for id in video_ids:
                    v_ids.append(id.cpu().numpy())
                for ques_id in question_ids:
                    q_ids.append(ques_id)

            total_acc += agreeings.float().sum().item()
            count += answers.size(0)
            print('avg_acc=',total_acc/count)
        acc = total_acc / count
        print('train set size:',count)
        print('acc on trainset:',acc)

    if not write_preds:
        return acc
    else:
        return acc, all_preds, gts, v_ids, q_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='sutd-traffic_transition.yml', type=str)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    assert cfg.dataset.name in ['sutd-traffic']
    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    temp_ckpt = "./results/sutd-traffic/ckpt/tempaligner_49.pt"
    semantic_ckpt = "./results/sutd-traffic/ckpt/semanticaligner_49.pt"

    ############################################################################ 
    ####uncomment the following lines to validate the trained model####
#    temp_ckpt = "./pretrained/tempaligner_49.pt"
#    semantic_ckpt = "./pretrained/semanticaligner_49.pt"
    ############################################################################

    assert os.path.exists(temp_ckpt)
    assert os.path.exists(semantic_ckpt)
    # load pretrained model
    loaded = torch.load(temp_ckpt, map_location='cpu')
    model_kwargs = loaded['model_kwargs']
    if cfg.dataset.name == 'sutd-traffic':
        cfg.dataset.annotation_file = cfg.dataset.annotation_file.format('test')
        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))

    else:
        pass

    test_loader_kwargs = {
        'appearance_feat': cfg.dataset.appearance_feat,
        'annotation_file': cfg.dataset.annotation_file,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'shuffle': False
    }
    test_loader = VideoQADataLoader(**test_loader_kwargs)

    tempaligner = TempAligner.TempAligner(**model_kwargs).to(device)
    model_dict = tempaligner.state_dict()
    state_dict = {k:v for k,v in loaded['state_dict'].items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    tempaligner.load_state_dict(model_dict)

    if cfg.test.write_preds:
        pass

    else:
        device = torch.device('cuda')
        loaded_semantic = torch.load(semantic_ckpt, map_location='cpu') 
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.float()
        semanticaligner = SemanticAligner().to(device)
        semanticaligner.load_state_dict(loaded_semantic['state_dict'])

        acc = validate(cfg, tempaligner, semanticaligner, clip_model, test_loader, device, cfg.test.write_preds)
        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()
