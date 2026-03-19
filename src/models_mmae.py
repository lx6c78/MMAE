import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_

import torch.nn.functional as F
from timm.models.vision_transformer import Mlp


class StrideEmbed(nn.Module):
    def __init__(self, img_height=40, img_width=40, stride_size=2, in_chans=1, embed_dim=192):              # 40  40
        super().__init__()
        self.num_patches = img_height * img_width // stride_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=stride_size, stride=stride_size)

    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
        return x

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Attention_losspred(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_stats_LT_guide=False, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        # self.attn = nn.MultiheadAttention(d_model, n_head)
        if self.fused_attn:
            if x_stats_LT_guide:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_bias,
                    dropout_p=self.attn_drop.p,
                )
            else:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p,
                )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if x_stats_LT_guide:
                if attn_bias is not None:
                    attn = attn + attn_bias

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 counter,
                 depth,
                 transformer_blocks,
                 num_patches,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if depth == 1:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
        elif counter in transformer_blocks:
            half_len = len(transformer_blocks) // 2
            if half_len == 0:
                self.mixer = Attention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    norm_layer=norm_layer,
                )
            else:
                first_half = set(transformer_blocks[:half_len])
                second_half = set(transformer_blocks[half_len:])
                if counter in first_half:
                    self.mixer = Attention(
                        dim,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=drop,
                        norm_layer=norm_layer,
                    )
                elif counter in second_half:
                    self.mixer = Attention(
                        dim,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=drop,
                        norm_layer=norm_layer,
                    )
        else:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Block_decoder(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x




class Block_losspred_decoder(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mixer = Attention_losspred(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x, x_stats_LT_guide=False, attn_bias=None):
        x = x + self.drop_path(
            self.gamma_1 * self.mixer(self.norm1(x), x_stats_LT_guide=x_stats_LT_guide, attn_bias=attn_bias))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# todo: class MambaVisionLayer(nn.Module): - > class MMAE_Layer(nn.Module)
class MMAE_Layer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 num_patches,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks=[],
                 ):

        super().__init__()

        self.blocks = nn.ModuleList([Block(dim=dim,
                                           counter=i,
                                           depth=depth,
                                           transformer_blocks=transformer_blocks,
                                           num_heads=num_heads,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias,
                                           num_patches=num_patches,
                                           qk_scale=qk_scale,
                                           drop=drop,
                                           attn_drop=attn_drop,
                                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                           layer_scale=layer_scale)
                                     for i in range(depth)])

    def forward(self, x):
        for _, blk in enumerate(self.blocks):
            x = blk(x)

        return x


# todo: class MambaVision(nn.Module): -> class MMAE_Core(nn.Module):
# todo: self.levels.append(level) 中的level = MambaVisionLayer -> level = MMAE_Layer
class MMAE_Core(nn.Module):
    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 num_patches,
                 drop_path_rate=0.2,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):

        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            # todo: self.levels.append(level) 中的level = MambaVisionLayer -> level = MMAE_Layer
            level = MMAE_Layer(dim=dim,
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     num_patches=num_patches,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < 3),
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[
                                                                                                          i] % 2 != 0 else list(
                                         range(depths[i] // 2, depths[i])),
                                     )
            self.levels.append(level)
        self.norm = nn.BatchNorm2d(num_features)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward(self, x):
        for level in self.levels:
            x = level(x)
        return x


class StatEncoder(nn.Module):
    def __init__(self, input_dim=27, encoded_dim=64):
        super().__init__()

        self.bn_input = nn.BatchNorm1d(input_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AdaptiveMaxPool1d(output_size=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(32, encoded_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.encoded_dim = encoded_dim

    def forward(self, x):
        batch_size = x.shape[0]
        x_norm = self.bn_input(x)
        feat = x_norm.unsqueeze(1)
        feat = self.cnn(feat)
        feat = feat.view(batch_size, -1)
        feat_deep = self.fc(feat)
        out = torch.cat([x_norm, feat_deep], dim=1)

        return out


# todo: class DualStreamStatRouter(nn.Module): -> class PMP(nn.Module):
class PMP(nn.Module):
    def __init__(self,
                 packet_size=320,
                 num_packets=5,
                 total_length=1600,
                 stride_size=4,
                 stat_embed_dim=64,
                 router_rank=16,
                 router_hidden=64,
                 gate_rank=32,
                 num_heads=8):
        super().__init__()

        self.packet_size = packet_size
        self.total_length = total_length
        self.num_heads = num_heads
        self.router_rank = router_rank

        self.time_embedder = StrideEmbed(img_height=40, img_width=40, stride_size=stride_size, in_chans=1,
                                         embed_dim=stat_embed_dim)

        self.len_embedder = StrideEmbed(img_height=40, img_width=40, stride_size=stride_size, in_chans=1,
                                        embed_dim=stat_embed_dim)

        self.num_patches = self.time_embedder.num_patches

        self.time_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, stat_embed_dim))
        self.len_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, stat_embed_dim))

        fusion_dim = stat_embed_dim * 2
        self.stat_fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )

        fusion_dim = stat_embed_dim * 2

        self.stat_fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        self.router_phi = nn.Sequential(
            nn.Linear(fusion_dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, router_rank),
        )
        self.router_U_sup = nn.Linear(router_rank, router_rank, bias=False)
        self.router_V_sup = nn.Linear(router_rank, router_rank, bias=False)
        self.router_U_del = nn.Linear(router_rank, router_rank, bias=False)
        self.router_V_del = nn.Linear(router_rank, router_rank, bias=False)
        self.stat_to_gate = nn.Linear(fusion_dim, gate_rank // 2)
        self.norm_to_saliency = nn.Linear(1, gate_rank // 4)

        input_dim = 1 + (gate_rank // 2) + (gate_rank // 4)
        self.selector_a = nn.Linear(input_dim, gate_rank, bias=False)
        self.selector_b = nn.Linear(input_dim, gate_rank, bias=False)
        self.router_gamma_sup = nn.Parameter(torch.tensor(0.0))
        self.router_gamma_del = nn.Parameter(torch.tensor(0.0))

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        trunc_normal_(self.time_pos_embed, std=.02)
        trunc_normal_(self.len_pos_embed, std=.02)

        w1 = self.time_embedder.proj.weight.data
        w2 = self.len_embedder.proj.weight.data

        torch.nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        torch.nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))

        for m in self.time_embedder.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.len_embedder.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        for m in self.stat_fusion:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.router_phi:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for lin in [self.router_U_sup, self.router_V_sup,
                    self.router_U_del, self.router_V_del]:
            nn.init.xavier_uniform_(lin.weight)

        for lin in [self.selector_a, self.selector_b,
                    self.stat_to_gate, self.norm_to_saliency]:
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.constant_(lin.bias, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, query, stats):
        B = query.shape[0]
        scale = self.router_rank ** -0.5

        if stats.dim() == 3:
            stats = stats.squeeze(1)

        times_val = stats[:, :5]
        lens_val = stats[:, 5:]
        raw_time_signal = times_val.unsqueeze(-1).repeat(1, 1, self.packet_size).reshape(B, 1, self.total_length)
        raw_len_signal = lens_val.unsqueeze(-1).repeat(1, 1, self.packet_size).reshape(B, 1, self.total_length)

        time_tokens = self.time_embedder(raw_time_signal) + self.time_pos_embed
        len_tokens = self.len_embedder(raw_len_signal) + self.len_pos_embed

        merged_stats = torch.cat([time_tokens, len_tokens], dim=-1)
        merged_stats = self.stat_fusion(merged_stats)

        z = self.router_phi(merged_stats)

        u_sup = self.router_U_sup(z)
        v_sup = self.router_V_sup(z)

        u_del = self.router_U_del(z)
        v_del = self.router_V_del(z)

        delta_sup = torch.einsum("bnr,bmr->bnm", u_sup, v_sup) * scale
        delta_del = torch.einsum("bnr,bmr->bnm", u_del, v_del) * scale

        with torch.no_grad():
            qn = F.normalize(query, dim=-1)
            sim_pos = torch.relu(torch.einsum("bne,bme->bnm", qn, qn))
            sim_pos = sim_pos - torch.diag_embed(torch.diagonal(sim_pos, dim1=1, dim2=2))
            similarity_feat = sim_pos.mean(dim=-1, keepdim=True)

        token_norm = torch.norm(query, p=2, dim=-1, keepdim=True)
        saliency_feat = self.norm_to_saliency(token_norm)


        stat_ctx = self.stat_to_gate(merged_stats)

        gate_input = torch.cat([similarity_feat, stat_ctx, saliency_feat], dim=-1)
        gate_logits = torch.einsum("bnr,bmr->bnm", self.selector_a(gate_input), self.selector_b(gate_input))

        p_sup = torch.sigmoid(gate_logits)
        p_del = 1.0 - p_sup

        gamma_sup = -F.softplus(self.router_gamma_sup)
        gamma_del = F.softplus(self.router_gamma_del)

        bias = p_sup * (gamma_sup * delta_sup) + p_del * (gamma_del * delta_del)  # [B, N, N]
        bias = bias.clamp(-5, 5)

        return bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

# todo: class NetMamba(nn.Module): -> class MMAE(nn.Module):
# todo: class NetMamba(nn.Module): 中的 self.mambavision -> self.mmae_core_block
# todo: self.mambavision = MambaVision -> self.mmae_core_block = MMAE_Core
# todo: class NetMamba(nn.Module): 中的 def mix_forward_statrouter_decoder - > def mix_forward_PMP_decoder
# todo: self.loss_predictor_router = DualStreamStatRouter - > self.loss_predictor_router = PMP
class MMAE(nn.Module):
    def __init__(self, img_size=40, stride_size=4, in_chans=1,
                 dim=256,
                 embed_dim=256,
                 decoder_embed_dim=128, decoder_depth=2,
                 decoder_num_heads=8,  # todo
                 num_classes=1000,
                 norm_pix_loss=False,
                 drop_path_rate=0.1,
                 is_pretrain=False,
                 teach_train=False,
                 learning_loss=True,
                 device=None, dtype=None,
                 **kwargs):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.is_pretrain = is_pretrain
        self.teach_train = teach_train
        self.learning_loss = learning_loss

        self.patch_embed = StrideEmbed(img_size, img_size, stride_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_cls_token = 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + num_cls_token, embed_dim))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # todo: class NetMamba(nn.Module): 中的 self.mambavision -> self.mmae_core_block
        self.mmae_core_block = MMAE_Core(depths=[1, 2, 2, 2],
                                       num_heads=[8, 8, 8, 8],
                                       window_size=[8, 8, 14, 7],
                                       dim=256,
                                       num_patches=self.num_patches,
                                       in_dim=32,
                                       mlp_ratio=4,
                                       drop_path_rate=0.2,
                                       **kwargs)
        if is_pretrain:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + num_cls_token, decoder_embed_dim))
            decoder_dpr = [x.item() for x in
                           torch.linspace(0, drop_path_rate, decoder_depth)]  # stochastic depth decay rule
            decoder_inter_dpr = [0.0] + decoder_dpr

            self.decoder_blocks = nn.ModuleList([
                Block_decoder(decoder_embed_dim, decoder_num_heads, mlp_ratio=4, qkv_bias=True, qk_scale=False,
                              norm_layer=nn.LayerNorm)
                for i in range(decoder_depth)])

            self.decoder_norm_f = None
            self.decoder_pred = nn.Linear(decoder_embed_dim, stride_size * in_chans, bias=True)  # decoder to stride

            self.stat_encoder = StatEncoder(input_dim=27, encoded_dim=64)
            if self.learning_loss:
                self.decoder_blocks_losspred = nn.ModuleList([
                    Block_losspred_decoder(decoder_embed_dim, decoder_num_heads, mlp_ratio=4, qkv_bias=True,
                                           qk_scale=False,
                                           norm_layer=nn.LayerNorm)
                    for i in range(decoder_depth)])
                self.decoder_norm_losspred = nn.LayerNorm(decoder_embed_dim)
                self.decoder_pred_losspred = nn.Linear(decoder_embed_dim, stride_size * in_chans, bias=True)
                # todo: self.loss_predictor_router = DualStreamStatRouter - > self.loss_predictor_router = PMP
                self.loss_predictor_router = PMP(packet_size=320, num_packets=5, total_length=1600,
                                                                  stride_size=stride_size, num_heads=decoder_num_heads)

        else:
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.norm_pix_loss = norm_pix_loss
        self.decoder_pred = nn.Linear(decoder_embed_dim, stride_size * in_chans, bias=True)  # decoder to stride
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        trunc_normal_(self.pos_embed, std=.02)
        if self.is_pretrain:
            trunc_normal_(self.decoder_pos_embed, std=.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        if self.is_pretrain:
            torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    def stride_patchify(self, imgs, stride_size=4):
        B, C, L = imgs.shape
        assert C == 1, "Input images should be grayscale"
        x = imgs.reshape(B, L // stride_size, stride_size)
        return x



    def select_generate_mask(self, x, loss_pred, mask_ratio=0.5, guide=True, epoch=0, total_epoch=88):
        B, L, D = x.shape

        len_keep = int(L * (1 - mask_ratio))
        len_mask = L - len_keep

        noise = torch.rand(B, L, device=x.device)

        if guide and loss_pred is not None and epoch > 0:
            loss_pred = loss_pred.squeeze()  # [B, L]

            max_hard_ratio = 1.0
            current_hard_ratio = max_hard_ratio * (epoch / total_epoch)
            len_hard_mask = int(len_mask * current_hard_ratio)

            if len_hard_mask > 0:
                topk_indices = torch.topk(loss_pred, k=len_hard_mask, dim=1).indices  # [B, len_hard_mask]
                noise.scatter_(1, topk_indices, noise.gather(1, topk_indices) + 10.0)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.unsqueeze(-1)  # [B, L, 1]

        return mask, ids_restore


    def get_symmetric_pairing(self, stats_emb):
        B = stats_emb.size(0)
        device = stats_emb.device

        stats_norm = F.normalize(stats_emb, p=2, dim=1)
        sim_matrix = stats_norm @ stats_norm.t()
        sim_matrix.fill_diagonal_(-float('inf'))
        pair_idx = torch.arange(B, device=device)
        current_sim = sim_matrix.clone()
        for _ in range(B // 2):
            max_val_idx = torch.argmax(current_sim)
            if current_sim.view(-1)[max_val_idx] == -float('inf'):
                break

            row = max_val_idx // B
            col = max_val_idx % B

            pair_idx[row] = col
            pair_idx[col] = row

            current_sim[row, :] = -float('inf')
            current_sim[:, row] = -float('inf')
            current_sim[col, :] = -float('inf')
            current_sim[:, col] = -float('inf')

        return pair_idx



    def mix_forward_encoder(self, x_payload, mask_ratio, x_stats=None, if_mask=True, gene_mask=None,
                            gene_ids_restore=None, loss_pred=None):

        B, C, L = x_payload.shape

        x = self.patch_embed(x_payload.reshape(B, C, L))

        if if_mask:
            assert x_stats is not None
            stats_enhanced = self.stat_encoder(x_stats)
            with torch.no_grad():
                perm_idx = self.get_symmetric_pairing(stats_enhanced)

            if gene_mask is not None:
                gene_mask = gene_mask.unsqueeze(-1)
                x = x * (1. - gene_mask) + x[perm_idx] * gene_mask

            else:
                mask, ids_restore = self.select_generate_mask(x, loss_pred, mask_ratio)
                x = x * (1. - mask) + x[perm_idx] * mask

        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, 0, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        hidden_states = x
        x = self.mmae_core_block(hidden_states)

        if if_mask:
            if gene_mask is not None:
                return x, gene_mask, gene_ids_restore, perm_idx
            else:
                return x, mask, ids_restore, perm_idx
        else:
            return x


    def mix_forward_decoder(self, x, mask, ids_restore, sample_inx, teach_train=False):
        # embed tokens

        x = self.decoder_embed(x)
        B, L, C = x.shape

        if teach_train == False:
            mask_tokens = self.mask_token.repeat(x.shape[0], L - 1, 1)

            x = x[:, 1:, :]
            cls_token = x[:, :1, :]
            x1 = x * (1 - mask) + mask_tokens * mask
            x1 = torch.cat([cls_token, x1], dim=1)  # append cls token

            x2 = x * mask + mask_tokens * (1 - mask)
            x2 = torch.cat([cls_token, x2], dim=1)  # append cls token

            x = torch.cat([x1, x2], dim=0)
        # add pos embed
        x = x + self.decoder_pos_embed

        loss_pred = x.clone()[:, 1:, :]

        hidden_states = x
        for blk in self.decoder_blocks:
            hidden_states = blk(hidden_states)
        x = self.decoder_norm(hidden_states)
        x_ = x[:, 1:, :]

        B, L, C = x_.shape
        if teach_train == False:

            x1_rec = x_[:B // 2]
            x2_rec = x_[B // 2:]

            inv_perm_idx = torch.argsort(sample_inx)
            x2_aligned = x2_rec[inv_perm_idx]
            pre_pred_unmix_x_rec = x1_rec * mask + x2_aligned * (1 - mask)
        else:
            pre_pred_unmix_x_rec = x_  # B= B
        # predictor projection
        x = self.decoder_pred(x)

        x = x[:, 1:, :]

        if self.learning_loss:
            for blk in self.decoder_blocks_losspred:
                loss_pred = blk(loss_pred)
            if teach_train == False:
                # unmix tokens
                x1_rec = loss_pred[:B // 2]
                x2_rec = loss_pred[B // 2:]
                inv_perm_idx = torch.argsort(sample_inx)
                x2_aligned = x2_rec[inv_perm_idx]
                loss_pred_unmix = x1_rec * mask + x2_aligned * (1 - mask)
            else:
                loss_pred_unmix = loss_pred  # B= B
            loss_pred_unmix = self.decoder_norm_losspred(loss_pred_unmix)
            loss_pred_unmix = self.decoder_pred_losspred(loss_pred_unmix)

            return x, pre_pred_unmix_x_rec, loss_pred_unmix.mean(dim=-1)  #

        return x, pre_pred_unmix_x_rec

    # todo: def mix_forward_statrouter_decoder - > def mix_forward_PMP_decoder
    def mix_forward_PMP_decoder(self, x, mask, ids_restore, sample_inx, teach_train=False, x_stats_LT=None):


        x = self.decoder_embed(x)
        B, L, C = x.shape

        if teach_train == False:

            mask_tokens = self.mask_token.repeat(x.shape[0], L - 1, 1)  # ([128, 400, 128])
            x = x[:, 1:, :]
            cls_token = x[:, :1, :]

            x1 = x * (1 - mask) + mask_tokens * mask
            x1 = torch.cat([cls_token, x1], dim=1)  # append cls token

            x2 = x * mask + mask_tokens * (1 - mask)
            x2 = torch.cat([cls_token, x2], dim=1)  # append cls token

            x = torch.cat([x1, x2], dim=0)
        # add pos embed
        x = x + self.decoder_pos_embed

        loss_pred = x.clone()[:, 1:, :]

        hidden_states = x
        for blk in self.decoder_blocks:
            hidden_states = blk(hidden_states)
        x = self.decoder_norm(hidden_states)

        x_ = x[:, 1:, :]

        B, L, C = x_.shape
        if teach_train == False:
            # unmix tokens
            x1_rec = x_[:B // 2]
            x2_rec = x_[B // 2:]
            inv_perm_idx = torch.argsort(sample_inx)
            x2_aligned = x2_rec[inv_perm_idx]
            pre_pred_unmix_x_rec = x1_rec * mask + x2_aligned * (1 - mask)
        else:
            pre_pred_unmix_x_rec = x_  # B= B
        x = self.decoder_pred(x)

        x = x[:, 1:, :]

        if self.learning_loss:
            stats_2B = torch.cat([x_stats_LT, x_stats_LT], dim=0)
            attn_bias = self.loss_predictor_router(query=x, stats=stats_2B)
            for blk in self.decoder_blocks_losspred:
                loss_pred = blk(loss_pred, x_stats_LT_guide=True, attn_bias=attn_bias)
            if teach_train == False:
                x1_rec = loss_pred[:B // 2]
                x2_rec = loss_pred[B // 2:]
                inv_perm_idx = torch.argsort(sample_inx)
                x2_aligned = x2_rec[inv_perm_idx]
                loss_pred_unmix = x1_rec * mask + x2_aligned * (1 - mask)
            else:
                loss_pred_unmix = loss_pred  # B= B
            loss_pred_unmix = self.decoder_norm_losspred(loss_pred_unmix)
            loss_pred_unmix = self.decoder_pred_losspred(loss_pred_unmix)

            return x, pre_pred_unmix_x_rec, loss_pred_unmix.mean(dim=-1)  #

        return x, pre_pred_unmix_x_rec


    def mix_forward_rec_loss(self, x, x_rec, mask, sample_inx):
        target = self.stride_patchify(x)
        B, L, C = x_rec.shape

        x1_rec = x_rec[:B // 2]
        x2_rec = x_rec[B // 2:]

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        inv_perm_idx = torch.argsort(sample_inx)
        x2_aligned = x2_rec[inv_perm_idx]
        unmix_x_rec = x1_rec * mask + x2_aligned * (1 - mask)
        loss = (unmix_x_rec - target) ** 2  # (B,L,C)
        loss_rec = loss.mean()  # (1,)
        loss_matrix = loss.mean(dim=-1)

        return loss_rec, loss_matrix, unmix_x_rec


    def forward(self, imgs, mask_ratio=0.5, gene_mask=None, gene_ids_restore=None, loss_pred=None):
        B, C, L = imgs.shape
        assert C == 1, "Input images should be grayscale"
        x_payload = imgs[:, :, :1600]  # Shape: [B, 1, 1600]

        x_stats = imgs[:, :, 1600:1627].squeeze(1)

        x_stats_LT = imgs[:, :, 1627:]

        if self.is_pretrain:
            if self.teach_train:
                x = self.mix_forward_encoder(x_payload, mask_ratio=mask_ratio, x_stats=None, if_mask=False,
                                             loss_pred=None)
                if self.learning_loss:
                    teach_output, pre_pred_unmix_x_rec, loss_pred = self.mix_forward_PMP_decoder(x, None, None,
                                                                                                        sample_inx=None,
                                                                                                        teach_train=True,
                                                                                                        x_stats_LT=x_stats_LT)
                    return teach_output, pre_pred_unmix_x_rec, loss_pred
                else:
                    teach_output, pre_pred_unmix_x_rec = self.mix_forward_decoder(x, None, None, sample_inx=None,
                                                                                  teach_train=True)
                    return teach_output, pre_pred_unmix_x_rec
            else:
                latent, mask, ids_restore, perm_idx = self.mix_forward_encoder(x_payload, mask_ratio=mask_ratio,
                                                                               x_stats=x_stats, if_mask=True,
                                                                               gene_mask=gene_mask,
                                                                               gene_ids_restore=gene_ids_restore,
                                                                               loss_pred=loss_pred)
                if self.learning_loss:
                    pred, pre_pred_unmix_x_rec, loss_pred = self.mix_forward_PMP_decoder(latent, mask,
                                                                                                ids_restore,
                                                                                                sample_inx=perm_idx,
                                                                                                x_stats_LT=x_stats_LT)
                    loss, loss_matrix, unmix_x_rec = self.mix_forward_rec_loss(x_payload, pred, mask,
                                                                               sample_inx=perm_idx)
                    return loss, loss_matrix, pred, mask, unmix_x_rec, pre_pred_unmix_x_rec, loss_pred, mask_ratio
                else:
                    pred, pre_pred_unmix_x_rec = self.mix_forward_decoder(latent, mask, ids_restore,
                                                                          sample_inx=perm_idx)
                    loss, loss_matrix, unmix_x_rec = self.mix_forward_rec_loss(x_payload, pred, mask,
                                                                               sample_inx=perm_idx)
                    return loss, loss_matrix, pred, mask, unmix_x_rec, pre_pred_unmix_x_rec, mask_ratio
        else:
            x = self.mix_forward_encoder(x_payload, mask_ratio=mask_ratio, x_stats=None, if_mask=False)
            return self.head(x[:, 0, :])

# todo: def net_mamba_pretrain(**kwargs): -> def mmae_pretrain(**kwargs):
def mmae_pretrain(**kwargs):
    model = MMAE(
        img_size=40, stride_size=4, in_chans=1,
        dim=256, depth=6,
        decoder_embed_dim=128, decoder_depth=2,
        drop_path_rate=0.1,
        is_pretrain=True,
        device=None, dtype=None,
        **kwargs)
    return model

# todo: def net_mamba_classifier(**kwargs): -> def mmae_classifier(**kwargs):
def mmae_classifier(**kwargs):
    model = MMAE(
        img_size=40, stride_size=4, in_chans=1,
        dim=256, depth=6,
        decoder_embed_dim=128, decoder_depth=2,
        is_pretrain=False, teach_train=False,
        device=None, dtype=None,
        **kwargs)
    return model

def mmae_teacher(**kwargs):
    model = MMAE(
        img_size=40, stride_size=4, in_chans=1,
        dim=256, depth=6,
        decoder_embed_dim=128, decoder_depth=2,
        is_pretrain=True, teach_train=True,
        device=None, dtype=None,
        **kwargs)
    return model