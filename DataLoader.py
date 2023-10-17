import json
import pickle
import torch
import math
import h5py
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
import time
from numpy import asarray
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np

def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['question_answer_idx_to_token'] = invert_dict(vocab['question_answer_token_to_idx'])
    return vocab





    



class VideoQADataset(Dataset):

    def __init__(self, answers, ans_candidates, questions, app_feature_h5, video_ids,
                 app_feat_id_to_index):
        self.all_answers = answers
        self.all_questions = questions
        self.all_video_ids = torch.LongTensor(np.asarray(video_ids))
        self.app_feature_h5 = app_feature_h5
        self.app_feat_id_to_index = app_feat_id_to_index
        self.all_ans_candidates = ans_candidates

    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        ans_candidates = self.all_ans_candidates[index]
        question = self.all_questions[index]
        video_idx = self.all_video_ids[index].item()
        app_index = self.app_feat_id_to_index[str(video_idx)]
        question_text = clip.tokenize(question) 
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in ans_candidates])

        with h5py.File(self.app_feature_h5, 'r') as f_app:
            appearance_feat = f_app['appearance_features'][app_index]  # (8, 16, 2048)

        appearance_feat = torch.from_numpy(appearance_feat)
        return (
            video_idx, answer, tokenized_prompts, appearance_feat, question_text,
        )

    def __len__(self):
        return len(self.all_questions)


class VideoQADataLoader(DataLoader):

    def __init__(self, **kwargs):
        annotation_file = kwargs.pop('annotation_file')

        with open(annotation_file) as f:
            instances = f.readlines()
        _header = instances.pop(0)
        questions = []
        answers = []
        video_names = []
        video_ids = []
        ans_candidates = []

        for instance in instances:
            data = json.loads(instance.strip())
            vid_id = data[1]
            video_ids.append(vid_id)
            vid_filename = data[2]
            video_names.append(vid_filename)
            q_body = data[4]
            questions.append(q_body)
            options = data[6:10]
            candidate = np.asarray( [ options[0], options[1], options[2], options[3] ] )
            ans_candidates.append( candidate )
            answer_idx = data[10]
            answers.append(answer_idx)
        print('number of questions: %s' % len(questions))

        with h5py.File(kwargs['appearance_feat'], 'r') as app_features_file:
            app_video_ids = app_features_file['ids'][()]

        app_feat_id_to_index = {str(id): i for i, id in enumerate(app_video_ids)}

        self.app_feature_h5 = kwargs.pop('appearance_feat')
        self.dataset = VideoQADataset(answers, ans_candidates, questions, self.app_feature_h5, video_ids,
                                      app_feat_id_to_index, 
                                      )
       
        self.batch_size = kwargs['batch_size']

        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
