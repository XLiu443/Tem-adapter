import numpy as np
from torch.nn import functional as F

from .utils import *




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TempAligner(nn.Module):
    def __init__(self, vision_dim, module_dim):
        super(TempAligner, self).__init__()

        self.positional_encoding = PositionalEncoding(module_dim, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=module_dim, nhead=16)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=module_dim, nhead=16) 
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        init_modules(self.modules(), w_init="xavier_uniform")

    def forward(self, answers, ans_candidates, video_appearance_feat, question,
                ):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question.size(0)

        correct_answers = []
        for ind in range(batch_size):
            batch_answer = ans_candidates[ind]
            correct_answer = batch_answer[answers[ind]]
            correct_answers.append(correct_answer)
        correct_answers = torch.stack(correct_answers)
        correct_answers = correct_answers.unsqueeze(1)
        correct_answers = correct_answers.permute(1,0,2)
        question_embedding = question
        feat_dim = video_appearance_feat.shape[-1]
        video_appearance_feat = video_appearance_feat.view(batch_size, -1, feat_dim)
        _, nframes, _ = video_appearance_feat.shape
        video_appearance_feat = video_appearance_feat.permute(1,0,2)
        video_appearance_feat = self.positional_encoding(video_appearance_feat)

        tgt_mask = nn.Transformer().generate_square_subsequent_mask(video_appearance_feat.size()[0])
        tgt_mask = tgt_mask.to(video_appearance_feat.device)
        src_key_padding_mask = torch.ones((batch_size, nframes), dtype=bool, device=video_appearance_feat.device)
        tgt_key_padding_mask = src_key_padding_mask

        visual_embedding = self.transformer_encoder(src=video_appearance_feat, src_key_padding_mask=~src_key_padding_mask)
        visual_embedding_answer = visual_embedding + correct_answers
        visual_embedding_decoder = self.transformer_decoder(tgt=video_appearance_feat, memory=visual_embedding_answer, tgt_mask=tgt_mask, tgt_key_padding_mask=~tgt_key_padding_mask)
        visual_embedding = visual_embedding.permute(1,0,2)
        visual_embedding_decoder = visual_embedding_decoder.permute(1,0,2)

        visual_embedding = torch.mean(visual_embedding, dim=1, keepdim=True)
        out = torch.matmul(ans_candidates, visual_embedding.permute(0,2,1)).view(batch_size*4, -1)
        return out, visual_embedding_decoder

