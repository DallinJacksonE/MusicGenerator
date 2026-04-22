import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MIDITransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super(MIDITransformer, self).__init__()
        self.d_model = d_model

        self.pitch_embed = nn.Embedding(128, d_model)
        self.inst_embed = nn.Embedding(128, d_model)
        self.vel_proj = nn.Linear(1, d_model)
        self.dur_proj = nn.Linear(1, d_model)
        self.delta_proj = nn.Linear(1, d_model)  # 5th Dimension

        self.pos_encoder = PositionalEncoding(
            d_model, dropout, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers)

        self.pitch_head = nn.Linear(d_model, 128)
        self.inst_head = nn.Linear(d_model, 128)
        self.vel_head = nn.Linear(d_model, 1)
        self.dur_head = nn.Linear(d_model, 1)
        self.delta_head = nn.Linear(d_model, 1)  # 5th Dimension

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        batch_size, seq_len, _ = src.shape

        pitch = src[:, :, 0].long()
        velocity = src[:, :, 1].unsqueeze(-1)
        duration = src[:, :, 2].unsqueeze(-1)
        instrument = src[:, :, 3].long()
        delta = src[:, :, 4].unsqueeze(-1)

        x = self.pitch_embed(pitch) + \
            self.inst_embed(instrument) + \
            self.vel_proj(velocity) + \
            self.dur_proj(duration) + \
            self.delta_proj(delta)

        x = self.pos_encoder(x)
        mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        output = self.transformer_encoder(x, mask=mask, is_causal=True)

        out_delta = self.delta_head(output).squeeze(-1)

        return {
            'pitch': self.pitch_head(output),
            'velocity': self.vel_head(output).squeeze(-1),
            'duration': self.dur_head(output).squeeze(-1),
            'instrument': self.inst_head(output),
            'delta': out_delta
        }
