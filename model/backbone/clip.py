# from https://github.com/openai/CLIP/blob/main/clip/model.py
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..pet_mixin import AdapterMixin, PrefixMixin, PromptMixin
from model.clip import clip


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# implementation from https://github.com/gqk/LAE/blob/master/libml/model/backbone/vit.py
class Attention(nn.Module, PromptMixin, PrefixMixin, AdapterMixin):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_mask=None,    # for text encoder
        prompt_per_task = 0,   # num of prompts for each task
        num_tasks = None  # num of tasks
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_mask = attn_mask
        self.prompt_per_task = prompt_per_task
        self.num_tasks = num_tasks

    def forward(self, x):
        x = self.add_prompt(x)  # not implementation

        N, B, C = x.shape       # num_patch, batch_size, dim
        
        qkv = self.adapt_module("qkv", x)

        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.chunk(3, dim=-1)
        k, v = self.add_prefix(k, v)

        q = q.contiguous().view(-1, B * self.num_heads, C // self.num_heads).transpose(0, 1)
        k = k.contiguous().view(-1, B * self.num_heads, C // self.num_heads).transpose(0, 1)
        v = v.contiguous().view(-1, B * self.num_heads, C // self.num_heads).transpose(0, 1)

        if self.attn_mask is not None:
            # text
            attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
            # visable
            prefix_attn_mask = torch.zeros(q.shape[1], k.shape[1]-attn_mask.shape[1]).type_as(attn_mask)
            attn_mask = torch.cat([prefix_attn_mask, attn_mask], dim=1)
            attn = torch.baddbmm(attn_mask, q * self.scale, k.transpose(-2, -1))  
        else:
            # image
            if self.prompt_per_task != 0:
                # attention mask
                attn_mask = torch.zeros(N, N).type_as(q)
                attn_mask[:, :(self.num_tasks+1)*self.prompt_per_task] = float("-inf")
                for i in range(self.num_tasks+1):
                    attn_mask[i*self.prompt_per_task:(self.num_tasks+1)*self.prompt_per_task, i*self.prompt_per_task:(i+1)*self.prompt_per_task] = 0
                attn = torch.baddbmm(attn_mask, q * self.scale, k.transpose(-2, -1))
            else:
                attn = torch.bmm(q * self.scale, k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = self.compensate_prefix(attn)

        x = torch.bmm(attn, v).transpose(0, 1).reshape(N, B, C)
        x = self.adapt_module("proj", x)  # x = self.proj(x)
        x = self.proj_drop(x)

        x = self.reduce_prompt(x)  # not implementation
        return x

class Mlp(nn.Module, AdapterMixin):
    def __init__(
        self,
        d_model,
    ):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        x = self.adapt_module("c_fc", x)  # x = self.c_fc(x)
        x = self.gelu(x)
        x = self.adapt_module("c_proj", x)  # x = self.c_proj(x)
        return x
    
class ResidualAttentionBlock(nn.Module, AdapterMixin):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, prompt_per_task:int = 0, num_tasks: int=None):
        super().__init__()

        # Modify the original Attention implementation
        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attn = Attention(d_model, n_head, qkv_bias=True, attn_mask=attn_mask, prompt_per_task=prompt_per_task, num_tasks=num_tasks)
        self.ln_1 = LayerNorm(d_model)
        # self.mlp = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(d_model * 4, d_model))
        # ]))
        self.mlp = Mlp(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def forward(self, x: torch.Tensor):
        # x = x + self.attn(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))

        x = x + self.adapt_module("attn", self.ln_1(x))
        x = x + self.adapt_module("mlp", self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompt_per_task: int = 0, num_tasks: int = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, prompt_per_task, num_tasks) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, prompt_per_task: int = 0, num_tasks: int=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, attn_mask=None, prompt_per_task=prompt_per_task, num_tasks=num_tasks)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.num_tasks = num_tasks

    def forward(self, x: torch.Tensor, task_prompt: bool=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # concat task prompt
        if task_prompt:
            assert len(self.task_prompt) >= self.num_tasks
            prompt = torch.cat([self.task_prompt[:self.num_tasks].detach().clone(), self.task_prompt[self.num_tasks:self.num_tasks+1]], dim=0)  # num_cur_tasks, prompt_per_task, width
            x = torch.cat([prompt.reshape(-1, self.transformer.width).unsqueeze(0).repeat(len(x), 1, 1), x], dim=1) # bs, num_cur_tasks * prompt_per_task + num_patchs + 1, width


        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD


        # split visual token
        if task_prompt:
            num_tokens = prompt.shape[0] * prompt.shape[1]
        else:
            num_tokens = 0


        x = self.ln_post(x[:, :num_tokens + 1, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,

                 # CoLeCLIP
                 init_checkpoint: str,
                 num_tasks: int,
                 prompt_per_task: int,
                 task_order: str
                 ################
                 ):
        super().__init__()

        self.context_length = context_length

        #CoLeCLIP
        self.init_checkpoint = init_checkpoint
        self.num_tasks = num_tasks
        self.prompt_per_task = prompt_per_task
        self.task_order = task_order

        self.vis_embed_dim = vision_width
        self.txt_embed_dim = transformer_width
        ###############################

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                prompt_per_task = prompt_per_task,
                num_tasks = num_tasks
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        if self.init_checkpoint is not None:
            self.load_pretrain()

    # add load CLIP weights
    def load_pretrain(self):
        checkpoint = torch.jit.load(self.init_checkpoint, map_location=torch.device('cpu')).state_dict()

        # rename the keys in checkpoint
        clip_weight_dict = {'in_proj_weight': 'qkv.weight',
                            'in_proj_bias': 'qkv.bias',
                            'out_proj.weight': 'proj.weight',
                            'out_proj.bias': 'proj.bias'}

        weight_key = list(checkpoint.keys())

        for k in weight_key:
            for clip_k, new_k in clip_weight_dict.items():
                if clip_k in k:
                    checkpoint[k.replace(clip_k, new_k)] = checkpoint.pop(k)
                    continue

        state = self.load_state_dict(checkpoint, strict=False)
        missing_keys, unexpected_keys = state
        assert missing_keys == []
        print(state)


    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.qkv.weight, std=attn_std)
            nn.init.normal_(block.attn.proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, task_prompt=False):
        # support task prompt learning
        return self.visual(image.type(self.dtype), task_prompt)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    
    @torch.no_grad()
    def zeroshot_classifier(self, cls_temp):
        # get text embeddings for all categories
        zeroshot_weights = {}

        for classnames, templates in cls_temp:
            if not isinstance(templates, list):
                templates = [templates]

            for classname in classnames:
                if classname in self.dup_cls_template:
                    dup_templates = self.dup_cls_template[classname]
                    texts = [template(classname) for template in dup_templates]  # format with class
                else:
                    texts = [template(classname) for template in templates]  # format with class
                texts = clip.tokenize(texts).cuda()  # tokenize

                class_embeddings = self.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights[classname] = class_embedding

        return zeroshot_weights
    
    @ torch.no_grad()
    def evaluate(self, images, text_features, cls_mapping):
        # predict
        image_features = self.encode_image(images).squeeze(1)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # fix bug from torch.norm() for consistency
        is_cur = torch.zeros(text_features.shape[1], dtype=torch.bool)
        is_cur[cls_mapping] = True
        text_features[:, is_cur] /= text_features[:, is_cur].norm(dim=0, keepdim=True)
        text_features[:, ~is_cur] /= text_features[:, ~is_cur].norm(dim=0, keepdim=True)

        # text_features /= text_features.norm(dim=0, keepdim=True)
        # fix bug for consistency
        logits = torch.zeros(len(images), text_features.shape[1]).type_as(images)
        logits[:, is_cur]  = self.logit_scale.exp() * image_features @ text_features[:, is_cur]
        logits[:, ~is_cur] = self.logit_scale.exp() * image_features @ text_features[:, ~is_cur]

        # logits = self.logit_scale.exp() * image_features @ text_features
        return logits

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


def ViT_B_16(cfg):
    assert cfg is not None

    if cfg.DATASETS.TRAIN != []:
        num_tasks = cfg.TASK_ORDER.index(cfg.DATASETS.TRAIN[0])
    else:
        assert cfg.TEST.CUR_TASK is not None
        num_tasks = cfg.TASK_ORDER.index(cfg.TEST.CUR_TASK)

    model_args = dict(
        embed_dim = 512, image_resolution = 224, vision_layers=12, 
        vision_width = 768, vision_patch_size = 16, context_length = 77,
        vocab_size = 49408, transformer_width = 512, transformer_heads = 8,
        transformer_layers = 12, prompt_per_task = cfg.METHOD.NUM_PROMPTS_PER_TASK,
        init_checkpoint = cfg.MODEL.VLM.LOAD, num_tasks = num_tasks,
        task_order = cfg.TASK_ORDER
    )
    model = CLIP(**model_args)
    return model


def ViT_L_14(cfg):
    assert cfg is not None

    if cfg.DATASETS.TRAIN != []:
        num_tasks = cfg.TASK_ORDER.index(cfg.DATASETS.TRAIN[0])
    else:
        assert cfg.TEST.CUR_TASK is not None
        num_tasks = cfg.TASK_ORDER.index(cfg.TEST.CUR_TASK)

    model_args = dict(
        embed_dim = 768, image_resolution = 224, vision_layers=24,  
        vision_width = 1024, vision_patch_size = 14, context_length = 77, 
        vocab_size = 49408, transformer_width = 768, transformer_heads = 12,
        transformer_layers = 12, prompt_per_task = cfg.METHOD.NUM_PROMPTS_PER_TASK,
        init_checkpoint = cfg.MODEL.VLM.LOAD, num_tasks = num_tasks,
        task_order = cfg.TASK_ORDER
    )
    model = CLIP(**model_args)
    return model


def build_vlm(cfg):
    if cfg.MODEL.VLM.NAME == 'ViT-B/16':
        return ViT_B_16(cfg)
    elif cfg.MODEL.VLM.NAME == 'ViT-L/14':
        return ViT_L_14(cfg)