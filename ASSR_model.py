import torch
from torch import nn
import numpy as np
from ASSR.time_lstm import Senti_Attenuation_LSTM

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class ASSR(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(ASSR, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # 情感的emb
        self.senti_bin_size = 100
        self.senti_pos_emb = torch.nn.Embedding(self.senti_bin_size + 1, args.hidden_units, padding_idx=0)
        self.senti_neg_emb = torch.nn.Embedding(self.senti_bin_size + 1, args.hidden_units, padding_idx=0)
        self.senti_con_emb = torch.nn.Embedding(self.senti_bin_size + 1, args.hidden_units, padding_idx=0)

        # to be Q for self-attention norm函数，自动学习均值和方差
        self.items_attention_layernorms = torch.nn.ModuleList()
        self.pos_attention_layernorms = torch.nn.ModuleList()
        self.neg_attention_layernorms = torch.nn.ModuleList()
        self.con_attention_layernorms = torch.nn.ModuleList()

        # self-attention函数
        self.items_attention_layers = torch.nn.ModuleList()
        self.pos_attention_layers = torch.nn.ModuleList()
        self.neg_attention_layers = torch.nn.ModuleList()
        self.con_attention_layers = torch.nn.ModuleList()

        # FNN的norm
        self.items_forward_layernorms = torch.nn.ModuleList()
        self.pos_forward_layernorms = torch.nn.ModuleList()
        self.neg_forward_layernorms = torch.nn.ModuleList()
        self.con_forward_layernorms = torch.nn.ModuleList()

        # FNN
        self.items_forward_layers = torch.nn.ModuleList()
        self.pos_forward_layers = torch.nn.ModuleList()
        self.neg_forward_layers = torch.nn.ModuleList()
        self.con_forward_layers = torch.nn.ModuleList()

        # 最终融合前的norm
        # 对张量的最后一维进行归一化，hidden_units=50，对列进行归一化
        self.items_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8).cuda()
        self.pos_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8).cuda()
        self.neg_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8).cuda()
        self.con_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8).cuda()

        # 情感衰减LSTM
        self.con_senti_lstm = Senti_Attenuation_LSTM(input_size=args.hidden_units, hidden_size=args.hidden_units).cuda()
        self.pos_senti_lstm = Senti_Attenuation_LSTM(input_size=args.hidden_units, hidden_size=args.hidden_units).cuda()
        self.neg_senti_lstm = Senti_Attenuation_LSTM(input_size=args.hidden_units, hidden_size=args.hidden_units).cuda()

        # 情感融合的norm
        self.senticoncat_norm = torch.nn.LayerNorm(args.hidden_units*3, eps=1e-8).cuda()
        # 情感融合的LSTM
        self.sentiment_lstm = torch.nn.LSTM(input_size=args.hidden_units*3, hidden_size=args.hidden_units, batch_first=True).cuda()

        self.pref_dense = torch.nn.Linear(args.hidden_units, args.hidden_units).cuda()

        for _ in range(args.num_blocks):
            # self-attention的Q归一化层
            attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            pos_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            neg_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            con_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

            # 添加归一化层
            self.items_attention_layernorms.append(attn_layernorm)
            self.pos_attention_layernorms.append(pos_layernorm)
            self.neg_attention_layernorms.append(neg_layernorm)
            self.con_attention_layernorms.append(con_layernorm)

            # attention_layers添加多头注意力层
            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            self.items_attention_layers.append(new_attn_layer)

            # pos的多头注意力层
            pos_attention = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            self.pos_attention_layers.append(pos_attention)

            # neg的多头注意力层
            neg_attention = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            self.neg_attention_layers.append(neg_attention)

            # con的多头注意力层
            con_attention = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            self.con_attention_layers.append(con_attention)

            # FNN的norm
            fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            pos_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            neg_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            con_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.items_forward_layernorms.append(fwd_layernorm)
            self.pos_forward_layernorms.append(pos_layernorm)
            self.neg_forward_layernorms.append(neg_layernorm)
            self.con_forward_layernorms.append(con_layernorm)

            # 添加FNN
            pos_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.pos_forward_layers.append(pos_fwd_layer)

            neg_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.neg_forward_layers.append(neg_fwd_layer)

            con_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.con_forward_layers.append(con_fwd_layer)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.items_forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs, con_sen_seqs, pos_sen_seqs, neg_sen_seqs, time_sen_seqs):

        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)
        tl = seqs.shape[1]
        seqs_attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        con_seqs = self.senti_con_emb(torch.LongTensor(con_sen_seqs).to(self.dev))
        con_seqs *= self.senti_con_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(con_sen_seqs.shape[1])), [con_sen_seqs.shape[0], 1])
        con_seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        con_seqs = self.emb_dropout(con_seqs)
        con_timeline_mask = torch.BoolTensor(con_sen_seqs == 0).to(self.dev)
        con_seqs *= ~con_timeline_mask.unsqueeze(-1)
        tl = con_seqs.shape[1]
        con_attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        pos_seqs = self.senti_pos_emb(torch.LongTensor(pos_sen_seqs).to(self.dev))
        pos_seqs *= self.senti_pos_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(pos_sen_seqs.shape[1])), [pos_sen_seqs.shape[0], 1])
        pos_seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        pos_seqs = self.emb_dropout(pos_seqs)
        pos_timeline_mask = torch.BoolTensor(pos_sen_seqs == 0).to(self.dev)
        pos_seqs *= ~pos_timeline_mask.unsqueeze(-1)
        tl = pos_seqs.shape[1]
        pos_attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        neg_seqs = self.senti_neg_emb(torch.LongTensor(neg_sen_seqs).to(self.dev))
        neg_seqs *= self.senti_neg_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(neg_sen_seqs.shape[1])), [neg_sen_seqs.shape[0], 1])
        neg_seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        neg_seqs = self.emb_dropout(neg_seqs)
        neg_timeline_mask = torch.BoolTensor(neg_sen_seqs == 0).to(self.dev)
        neg_seqs *= ~neg_timeline_mask.unsqueeze(-1)
        tl = neg_seqs.shape[1]
        neg_attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        new_con_seqs, (ht, ct) = self.con_senti_lstm(con_seqs, time_sen_seqs)
        new_pos_seqs, (ht, ct) = self.pos_senti_lstm(pos_seqs, time_sen_seqs)
        new_neg_seqs, (ht, ct) = self.neg_senti_lstm(neg_seqs, time_sen_seqs)

        for i in range(len(self.items_attention_layers)):
            new_con_seqs = torch.transpose(new_con_seqs, 0, 1)
            new_con_Q = self.con_attention_layernorms[i](new_con_seqs)
            seqs = torch.transpose(seqs, 0, 1)
            mha_outputs, _ = self.con_attention_layers[i](new_con_Q, seqs, seqs, attn_mask=con_attention_mask)
            seqs = torch.transpose(seqs, 0, 1)
            new_con_seqs = new_con_Q + mha_outputs
            new_con_seqs = torch.transpose(new_con_seqs, 0, 1)
            new_con_seqs = self.con_forward_layernorms[i](new_con_seqs)
            new_con_seqs = self.con_forward_layers[i](new_con_seqs)
            new_con_seqs *= ~con_timeline_mask.unsqueeze(-1)
            new_con_seqs = self.con_last_layernorm(new_con_seqs)

            new_pos_seqs = torch.transpose(new_pos_seqs, 0, 1)
            new_pos_Q = self.pos_attention_layernorms[i](new_pos_seqs)
            seqs = torch.transpose(seqs, 0, 1)
            mha_outputs, _ = self.pos_attention_layers[i](new_pos_Q, seqs, seqs, attn_mask=pos_attention_mask)
            seqs = torch.transpose(seqs, 0, 1)
            new_pos_seqs = new_pos_Q + mha_outputs
            new_pos_seqs = torch.transpose(new_pos_seqs, 0, 1)
            new_pos_seqs = self.pos_forward_layernorms[i](new_pos_seqs)
            new_pos_seqs = self.pos_forward_layers[i](new_pos_seqs)
            new_pos_seqs *= ~pos_timeline_mask.unsqueeze(-1)
            new_pos_seqs = self.pos_last_layernorm(new_pos_seqs)

            new_neg_seqs = torch.transpose(new_neg_seqs, 0, 1)
            new_neg_Q = self.neg_attention_layernorms[i](new_neg_seqs)
            seqs = torch.transpose(seqs, 0, 1)
            mha_outputs, _ = self.neg_attention_layers[i](new_neg_Q, seqs, seqs, attn_mask=neg_attention_mask)
            seqs = torch.transpose(seqs, 0, 1)
            new_neg_seqs = new_neg_Q + mha_outputs
            new_neg_seqs = torch.transpose(new_neg_seqs, 0, 1)
            new_neg_seqs = self.neg_forward_layernorms[i](new_neg_seqs)
            new_neg_seqs = self.neg_forward_layers[i](new_neg_seqs)
            new_neg_seqs *= ~neg_timeline_mask.unsqueeze(-1)
            new_neg_seqs = self.neg_last_layernorm(new_neg_seqs)

            seqs = torch.transpose(seqs, 0, 1)
            seqs_Q = self.items_attention_layernorms[i](seqs)
            mha_outputs, _ = self.items_attention_layers[i](seqs_Q, seqs, seqs, attn_mask=seqs_attention_mask)
            seqs = seqs_Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.items_forward_layernorms[i](seqs)
            seqs = self.items_forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)
            seqs = self.items_last_layernorm(seqs)

        # AESGS-concat
        senti_concat = torch.cat((new_pos_seqs, new_neg_seqs, new_con_seqs), -1).cuda()
        senti_concat = self.senticoncat_norm(senti_concat)
        senti_pref_feats, (p_ge_ht, p_ge_ct) = self.sentiment_lstm(senti_concat)
        # senti_pref_feats = self.pref_dense(senti_pref_feats)
        log_feats = seqs + senti_pref_feats

        return log_feats

    def forward(self, user_ids, log_seqs, con_sen_seqs, pos_sen_seqs, neg_sen_seqs, time_sen_seq, pos_seqs, neg_seqs):
        # for training

        log_feats = self.log2feats(log_seqs, con_sen_seqs, pos_sen_seqs, neg_sen_seqs, time_sen_seq)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # 正样得分
        neg_logits = (log_feats * neg_embs).sum(dim=-1)  # 负样得分

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, con_sen_seqs, pos_sen_seqs, neg_sen_seqs, time_sen_seq, item_indices):
        # for inference
        log_feats = self.log2feats(log_seqs, con_sen_seqs, pos_sen_seqs, neg_sen_seqs, time_sen_seq)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
