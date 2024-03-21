
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.utils.model_ema import ModelEmaV2
from model.backbone.clip import build_vlm
from model.pet import Adapter, KVLoRA, Prefix
from datasets.name_template import name_temp_info

class LAE(nn.Module):
    def __init__(self, cfg, eval):
        super(LAE, self).__init__()
        self.cfg = cfg

        # vlm 
        self.vlm = build_vlm(cfg)
        self.freeze(self.vlm)


        # PEFT for image encoder
        self.vis_pets, self.txt_pets = self.create_pets()
        assert self.txt_pets == None and self.vis_pets is not None
        # viusal
        if self.vis_pets is not None:
            print(f"==> visual pets:\n{self.vis_pets}")
            self.vis_pets_emas = nn.ModuleList([])
        
        # load checkpoints
        if not eval:    
            # PEFT   first task
            if cfg.DATASETS.TRAIN[0] in cfg.TASK_ORDER[:2]:
                self.load_checkpoint(filter='txt_pets')

            # instantiate ModelEmaV2 after the first task
            if cfg.DATASETS.TRAIN[0] != cfg.TASK_ORDER[0]:
                if len(self.vis_pets_emas) < self.cfg.METHOD.NUM_EMAS:
                    idx = len(self.vis_pets_emas)
                    ema = ModelEmaV2(self.vis_pets, decay=self.cfg.METHOD.EMA_DECAY)  
                    self.vis_pets_emas.append(ema)
                if cfg.DATASETS.TRAIN[0] not in cfg.TASK_ORDER[:2]:
                    self.load_checkpoint(filter='txt_pets')
        else:
            if cfg.TEST.CUR_TASK != cfg.TASK_ORDER[0]:
                if len(self.vis_pets_emas) < self.cfg.METHOD.NUM_EMAS:
                    idx = len(self.vis_pets_emas)
                    ema = ModelEmaV2(self.vis_pets, decay=self.cfg.METHOD.EMA_DECAY)  
                    self.vis_pets_emas.append(ema)
            self.load_checkpoint(filter='txt_pets')

        self.attach_pets(self.vis_pets, self.txt_pets)
                   
        # classifier
        # dictionary tuple:  {cls: (weights, bias)}
        self.classifier = torch.load(cfg.METHOD.VOCAB_PTH) if cfg.METHOD.VOCAB_PTH is not None else {}
        if cfg.METHOD.VOCAB_PTH is not None:
            print("Classifier loaded from", cfg.METHOD.VOCAB_PTH)
        
        # Initialize a new classifier
        if not eval:
            cur_cls = name_temp_info[cfg.DATASETS.TRAIN[0]][0]
            num_cls = len(cur_cls)
            weights = torch.zeros(self.vlm.visual.output_dim, num_cls) # dim, num_cls
            bias = torch.zeros(1, num_cls)  # 1, num_cls

            # init
            nn.init.normal_(weights, std=0.001)
            nn.init.constant_(bias, 0.0)

            # load from previous classifier
            for i, cls in enumerate(cur_cls):
                if cls in self.classifier:
                    weights[:, i], bias[:, i] = self.classifier[cls]
        
            self.classifier_weights = nn.Parameter(weights)  # dim, num_cls
            self.classifier_bias = nn.Parameter(bias)        # 1, num_cls


    def post_train_step(self):
        assert self.vis_pets is not None
        for idx, ema in enumerate(reversed(self.vis_pets_emas)):
            if idx == 0:  # the last one
                ema.update(self.vis_pets)
            else:
                ema.update(self.vis_pets_emas[idx - 1])
    
    def freeze_pets(self):
        for v in self.vis_pets.parameters():
            v.requires_grad = False
    
    def unfreeze_pets(self):
        for v in self.vis_pets.parameters():
            v.requires_grad = True

    def create_pets(self):
        # visual / text + peft
        vis_pets, txt_pets = None, None
        if self.cfg.METHOD.VIS:
            vis_pets = self.create_pets_clip('vis')
        if self.cfg.METHOD.TXT:
            txt_pets = self.create_pets_clip('txt')
        return vis_pets, txt_pets

    def create_pets_clip(self, encoder_type):
        assert self.cfg.METHOD.PET_CLS in ["Adapter", "LoRA", "Prefix"]

        n = len(self.cfg.METHOD.ADAPT_BLOCKS)

        if encoder_type == 'vis':
            embed_dim = self.vlm.vis_embed_dim 
        elif encoder_type == 'txt':
            embed_dim = self.vlm.txt_embed_dim 

        kwargs = dict(**self.cfg.METHOD.PET_KWARGS)
        if self.cfg.METHOD.PET_CLS == "Adapter":
            kwargs["embed_dim"] = embed_dim
            return nn.ModuleList([Adapter(**kwargs) for _ in range(n)])

        if self.cfg.METHOD.PET_CLS == "LoRA":
            kwargs["in_features"] = embed_dim
            kwargs["out_features"] = embed_dim
            return nn.ModuleList([KVLoRA(**kwargs) for _ in range(n)])

        kwargs["dim"] = embed_dim
        return nn.ModuleList([Prefix(**kwargs) for i in range(n)])
    
    def freeze(self, model):
        for k, v in model.named_parameters():
            v.requires_grad = False

    def attach_pets_clip(self, pets: nn.ModuleList, encoder_type: str):
        assert self.cfg.METHOD.PET_CLS in ["Adapter", "LoRA", "Prefix"]

        if self.cfg.METHOD.PET_CLS == "Adapter":
            for i, b in enumerate(self.cfg.METHOD.ADAPT_BLOCKS):
                if encoder_type == 'vis':
                   self.vlm.visual.transformer.resblocks[b].attach_adapter(attn=pets[i])

                elif encoder_type == 'txt':
                    self.vlm.transformer.resblocks[b].attach_adapter(attn=pets[i])
            return

        if self.cfg.METHOD.PET_CLS == "LoRA":
            for i, b in enumerate(self.cfg.METHOD.ADAPT_BLOCKS):
                if encoder_type == 'vis':
                        self.vlm.visual.transformer.resblocks[b].attn.attach_adapter(qkv=pets[i])
                elif encoder_type == 'txt':
                        self.vlm.transformer.resblocks[b].attn.attach_adapter(qkv=pets[i])
            return

        for i, b in enumerate(self.cfg.METHOD.ADAPT_BLOCKS):
            if encoder_type == 'vis':
                    self.vlm.visual.transformer.resblocks[b].attn.attach_prefix(pets[i])
            elif encoder_type == 'txt':
                    self.vlm.transformer.resblocks[b].attn.attach_prefix(pets[i])

    def attach_pets(self, vis_pets: nn.ModuleList, txt_pets: nn.ModuleList):
        assert txt_pets is None
        if vis_pets is not None:
            self.attach_pets_clip(vis_pets, 'vis')
        if txt_pets is not None:
            self.attach_pets_clip(txt_pets, 'txt')

    

    def forward(self, images, labels, texts, stage):
        logit = self.get_logit(images, self.classifier_weights, self.classifier_bias)
        loss = F.cross_entropy(logit, labels)
        return loss

    def get_logit(self, images, weights, bias):
        # extract image embedding
        devices = list(range(torch.cuda.device_count()))
        image_embed = nn.parallel.data_parallel(self.vlm.visual, images, device_ids=devices).squeeze(1)

        logit = image_embed @ weights + bias # bs, num_cls
        return logit

    @torch.no_grad()
    def evaluate(self, images, text_features, cls_mapping):
        self.eval()

        weights, bias = text_features

        # logit_on
        logit_on = self.get_logit(images, weights, bias) # bs, num_cls

        logit_emas = [logit_on]

        for ema in self.vis_pets_emas:
            self.attach_pets(ema.module, txt_pets=None)
            logit_emas.append(self.get_logit(images, weights, bias))

        self.attach_pets(self.vis_pets, self.txt_pets)
        cil_logit = torch.stack([logit.softmax(dim=1) for logit in logit_emas], dim=-1).max(dim=-1)[0]   # bs, num_cls
        til_logit = torch.stack([logit[:, cls_mapping].softmax(dim=1) for logit in logit_emas], dim=-1).max(dim=-1)[0]   # bs, num_cur_cls
        return (cil_logit, til_logit)


    # filter parameters which is not saved
    def filter_state_dict(self, state_dict):
        return {
            k: v
            for k, v in state_dict.items()
            if not k.startswith('vlm') and not 'classifier_weights' in k and not 'txt_pets' in k and not 'classifier_bias' in k and not 'classifier' in k
        }
    
    def save_checkpoint(self, save_pth):
        # save PEFT
        if os.path.dirname(save_pth) != "":
            os.makedirs(os.path.dirname(save_pth), exist_ok=True)
        
        state_dict = self.filter_state_dict(self.state_dict())

        torch.save({"state_dict": state_dict}, save_pth)
        print("Checkpoint saved to", save_pth)


    def load_checkpoint(self, filter='txt_pets'):
        # vlm + PEFT(image)
        # only load PEFT(image)
        if self.cfg.MODEL.LOAD is not None:
            checkpoint = torch.load(self.cfg.MODEL.LOAD)
        else:
            print("not load any checkpoints!")
            return
        
        # not load peft weights
        if filter is not None:
            checkpoint['state_dict'] =  {k:v  for k, v in checkpoint['state_dict'].items() if filter not in k}
        
        missing_keys, unexpected_keys = self.load_state_dict(
                checkpoint["state_dict"], strict=False
            )
        # filter vlm parameters
        missing_keys = [k for k in missing_keys if 'vlm' not in k]
        if filter is not None:
            missing_keys = [k for k in missing_keys if filter not in k]
        
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
        
        print("Checkpoint loaded from", self.cfg.MODEL.LOAD)




        

        

