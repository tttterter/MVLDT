import math
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F


class DyT(nn.Module):
    def __init__(self, dim, init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta


class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        Q = self.W_q(x).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        return output


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = DyT(d_model)
        self.norm2 = DyT(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                encoder_layer.self_attn.d_model,
                encoder_layer.self_attn.nhead
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDenoisingAutoEncoder(nn.Module):
    def __init__(self, seq_length=64, embed_dim=8, noise_level=0.3, num_layers=2, nhead=2):
        super(TransformerDenoisingAutoEncoder, self).__init__()
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.noise_level = noise_level

        self.yasuo = nn.Linear(256, seq_length)
        self.embedding = nn.Linear(1, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, embed_dim))

        encoder_layer = CustomTransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(embed_dim, 1)
        self.pengzhang = nn.Linear(seq_length, 256)

        self._init_weights()

    def forward(self, x):
        x = self.yasuo(x)
        x = F.relu(x)

        x_noisy = x + self.noise_level * torch.randn_like(x)
        x_noisy = x_noisy.unsqueeze(-1)

        x_embed = self.embedding(x_noisy)
        x_embed = F.relu(x_embed)

        x_embed = x_embed + self.pos_embedding
        x_encoded = self.transformer_encoder(x_embed)

        x_decoded = self.decoder(x_encoded)
        x_decoded = F.relu(x_decoded)

        output = x_decoded.squeeze(-1)
        output = self.pengzhang(output)
        output = torch.sigmoid(output)

        return output

    def _init_weights(self):
        nn.init.xavier_uniform_(self.yasuo.weight)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.xavier_uniform_(self.pengzhang.weight)


class CKAN(nn.Module):
    def __init__(self, args, n_entity, n_relation, emb_noise_level=0.3, n_views=3, dropout_rate=0.5,
                 fusion_type='attention'):
        super(CKAN, self).__init__()
        self._parse_args(args, n_entity, n_relation)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.scaler = StandardScaler()

        # Multi-view parameters
        self.n_views = n_views
        self.dropout_rate = dropout_rate
        self.fusion_type = fusion_type

        # Learnable attention for fusion if using attention strategy
        if fusion_type == 'attention':
            self.fusion_attention = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.ReLU(),
                nn.Linear(self.dim, 1),
                nn.Softmax(dim=1)
            )

        self.attention = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, 1, bias=False),
            nn.Sigmoid(),
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.user_dae = TransformerDenoisingAutoEncoder()
        self.item_dae = TransformerDenoisingAutoEncoder()

        self.W_Q_U = []
        for i in range(0, self.n_layer):
            x = nn.Parameter(torch.randn(self.dim, self.dim)).to(device)
            self.W_Q_U.append(x)

        self.W_K_U = []
        for i in range(0, self.n_layer):
            x = nn.Parameter(torch.randn(self.dim, self.dim)).to(device)
            self.W_K_U.append(x)

        self.W_Q_V = []
        for i in range(0, self.n_layer):
            x = nn.Parameter(torch.randn(self.dim, self.dim)).to(device)
            self.W_Q_V.append(x)

        self.W_K_V = []
        for i in range(0, self.n_layer):
            x = nn.Parameter(torch.randn(self.dim, self.dim)).to(device)
            self.W_K_V.append(x)

        self.ConVU = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(self.n_layer + 1, self.n_layer + 1))
        self.ConVV = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(self.n_layer + 1, self.n_layer + 1))
        self._init_weight()

    def _create_view(self, h_emb, r_emb, t_emb):
        """Create a single view by randomly dropping some knowledge"""
        mask = torch.bernoulli(torch.ones_like(h_emb[:, :, 0]) * (1 - self.dropout_rate))
        mask = mask.unsqueeze(-1).expand_as(h_emb)

        h_emb_view = h_emb * mask
        r_emb_view = r_emb * mask
        t_emb_view = t_emb * mask

        return self._knowledge_attention(h_emb_view, r_emb_view, t_emb_view)

    def _fuse_views(self, view_embeddings):
        """Fuse multiple views based on the specified strategy"""
        if self.fusion_type == 'mean':
            return torch.mean(view_embeddings, dim=1)
        elif self.fusion_type == 'max':
            return torch.max(view_embeddings, dim=1)[0]
        elif self.fusion_type == 'attention':
            # [batch_size, n_views, 1]
            attention_weights = self.fusion_attention(view_embeddings)
            # [batch_size, dim]
            fused = torch.sum(attention_weights * view_embeddings, dim=1)
            return fused
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

    def _multi_view_knowledge_attention(self, h_emb, r_emb, t_emb):
        """Process multiple views of the knowledge graph"""
        batch_size = h_emb.size(0)
        view_embeddings = []

        # Create multiple views
        for _ in range(self.n_views):
            view_emb = self._create_view(h_emb, r_emb, t_emb)
            view_embeddings.append(view_emb)

        # Stack views: [batch_size, n_views, dim]
        view_embeddings = torch.stack(view_embeddings, dim=1)

        # Fuse views
        fused_embedding = self._fuse_views(view_embeddings)
        return fused_embedding

    def forward(self, items, user_triple_set, item_triple_set):
        user_embeddings = []

        # Process user embeddings with multi-view
        h_emb = self.entity_emb(user_triple_set[0][0])
        r_emb = self.relation_emb(user_triple_set[1][0])
        t_emb = self.entity_emb(user_triple_set[2][0])
        user_emb_0 = self._multi_view_knowledge_attention(h_emb, r_emb, t_emb)
        noisy_user_emb_0 = self.user_dae(user_emb_0)
        user_embeddings.append(noisy_user_emb_0)

        User_emb_i = []
        Su = []

        for i in range(self.n_layer):
            h_emb = self.entity_emb(user_triple_set[0][i])
            r_emb = self.relation_emb(user_triple_set[1][i])
            t_emb = self.entity_emb(user_triple_set[2][i])
            user_emb_i = self._multi_view_knowledge_attention(h_emb, r_emb, t_emb)
            User_emb_i.append(user_emb_i)

            e_u_Q = torch.matmul(user_emb_0, self.W_Q_U[i])
            e_u_K = torch.matmul(user_emb_i, self.W_K_U[i])
            su = (e_u_Q * e_u_K).sum(dim=1)
            su = su / math.sqrt(self.dim)
            su = su.unsqueeze(1)
            su = F.softmax(su, dim=0)
            Su.append(su)

            user_emb_i = user_emb_i * su + user_emb_i
            user_embeddings.append(user_emb_i)

        # Process item embeddings
        item_embeddings = []
        item_emb_origin = self.entity_emb(items)
        noisy_item_emb_origin = self.item_dae(item_emb_origin)
        item_embeddings.append(item_emb_origin)

        Item_emb_i = []

        for i in range(self.n_layer):
            h_emb = self.entity_emb(item_triple_set[0][i])
            r_emb = self.relation_emb(item_triple_set[1][i])
            t_emb = self.entity_emb(item_triple_set[2][i])
            item_emb_i = self._multi_view_knowledge_attention(h_emb, r_emb, t_emb)
            Item_emb_i.append(item_emb_i)

            e_v_Q = torch.matmul(noisy_item_emb_origin, self.W_Q_V[i])
            e_v_K = torch.matmul(item_emb_i, self.W_K_V[i])
            sv = (e_v_Q * e_v_K).sum(dim=1)
            sv = sv / math.sqrt(self.dim)
            sv = sv.unsqueeze(1)
            sv = F.softmax(sv, dim=-1)
            item_emb_i = item_emb_i * sv + item_emb_i
            item_embeddings.append(item_emb_i)

        scores = self.predict2(user_embeddings, item_embeddings)

        # DAE Loss
        user_emb_reconstructed = user_emb_0
        user_dae_loss = F.mse_loss(user_emb_reconstructed, noisy_user_emb_0)

        item_emb_reconstructed = self.entity_emb(items)
        item_dae_loss = F.mse_loss(item_emb_reconstructed, noisy_item_emb_origin)

        return scores, user_dae_loss, item_dae_loss

    def predict2(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]
        e_u = torch.unsqueeze(e_u, 1)
        e_v = torch.unsqueeze(e_v, 1)

        for i in range(1, len(user_embeddings)):
            e = torch.unsqueeze(user_embeddings[i], 1)
            e_u = torch.cat((e, e_u), dim=1)

        for i in range(1, len(item_embeddings)):
            e = torch.unsqueeze(item_embeddings[i], 1)
            e_v = torch.cat((e, e_v), dim=1)

        e_u = torch.unsqueeze(e_u, 1)
        e_v = torch.unsqueeze(e_v, 1)

        u = self.ConVU(e_u)
        v = self.ConVV(e_v)

        u = torch.squeeze(u, 1)
        v = torch.squeeze(v, 1)
        u = torch.squeeze(u, 1)
        v = torch.squeeze(v, 1)

        scores = (v * u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores

    def predict(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]

        if self.agg == 'concat':
            if len(user_embeddings) != len(item_embeddings):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u), dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v), dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(user_embeddings)):
                e_u += user_embeddings[i]
            for i in range(1, len(item_embeddings)):
                e_v += item_embeddings[i]
        elif self.agg == 'pool':
            for i in range(1, len(user_embeddings)):
                e_u = torch.max(e_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                e_v = torch.max(e_v, item_embeddings[i])
        else:
            raise Exception("Unknown aggregator: " + self.agg)

        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores

    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.agg = args.agg

    def _init_weight(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.ConVU.weight)
        nn.init.xavier_uniform_(self.ConVV.weight)

        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        if self.fusion_type == 'attention':
            for layer in self.fusion_attention:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        att_weights = self.attention(torch.cat((h_emb, r_emb), dim=-1)).squeeze(-1)
        att_weights_norm = F.softmax(att_weights, dim=-1)
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)
        emb_i = emb_i.sum(dim=1)
        return emb_i