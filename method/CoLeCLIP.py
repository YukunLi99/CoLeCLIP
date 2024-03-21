import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.clip import build_vlm
from model.pet import Adapter, KVLoRA, Prefix
from datasets.name_template import name_temp_info


class CoLeCLIP(nn.Module):
    def __init__(self, cfg, eval):
        super(CoLeCLIP, self).__init__()
        self.cfg = cfg

        self.prompt_per_task = cfg.METHOD.NUM_PROMPTS_PER_TASK
        self.tol_num_tasks = len(cfg.TASK_ORDER)

        # vlm 
        self.vlm = build_vlm(cfg)
        self.freeze(self.vlm)

        # task prompt for image encoder
        if self.prompt_per_task != 0:
            self.vlm.visual.task_prompt = nn.Parameter(torch.zeros(self.tol_num_tasks, self.prompt_per_task, self.vlm.vis_embed_dim))
            nn.init.uniform_(self.vlm.visual.task_prompt.data, 0, 0.01)
        
        # Get the number of learned tasks
        self.num_tasks = cfg.TASK_ORDER.index(cfg.DATASETS.TRAIN[0])

        
        if not eval:
            # only for training time
            # PEFT for text encoder
            self.vis_pets, self.txt_pets = self.create_pets()
            assert self.vis_pets == None
            # text
            if self.txt_pets is not None:
                print(f"==> text pets:\n{self.txt_pets}")
            self.attach_pets(self.vis_pets, self.txt_pets)
        
        # load checkpoints for task prompt
        self.load_checkpoint(filter='txt_pets')

        # load dict which stores classes for each task
        self.task_class_dict = json.load(open(cfg.METHOD.TASK_CLS_DICT, 'r')) if cfg.METHOD.TASK_CLS_DICT is not None else {}
        if cfg.METHOD.TASK_CLS_DICT is not None:
            print("Task Class Dict loaded from", cfg.METHOD.TASK_CLS_DICT)
        
        # class vocabulary
        self.vocab = torch.load(cfg.METHOD.VOCAB_PTH) if cfg.METHOD.VOCAB_PTH is not None else {}
        if cfg.METHOD.VOCAB_PTH is not None:
            print("Class Vocabulary loaded from", cfg.METHOD.VOCAB_PTH)


    def freeze(self, model):
        for k, v in model.named_parameters():
            v.requires_grad = False
        
    def attach_pets(self, vis_pets: nn.ModuleList, txt_pets: nn.ModuleList):
        assert vis_pets is None
        if vis_pets is not None:
            self.attach_pets_clip(vis_pets, 'vis')
        if txt_pets is not None:
            self.attach_pets_clip(txt_pets, 'txt')
        


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
    
    def load_checkpoint(self, filter='txt_pets'):
        # vlm + PEFT(text) + task prompt(image)
        # only load task prompt
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
        missing_keys = [k for k in missing_keys if 'vlm' not in k or 'vlm.visual.task_prompt' in k]
        if filter is not None:
            missing_keys = [k for k in missing_keys if filter not in k]
        
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
        
        print("Checkpoint loaded from", self.cfg.MODEL.LOAD)


    def zeroshot_classifier(self, cls_temp):
       return self.vlm.zeroshot_classifier(cls_temp)

    
    # filter parameters which is not saved
    def filter_state_dict(self, state_dict):
        return {
            k: v
            for k, v in state_dict.items()
            if 'vlm.visual.task_prompt' in k
        }
    
    def save_checkpoint(self, save_pth):
        # only save class vocabulary & task class dict
        if os.path.dirname(save_pth) != "":
            os.makedirs(os.path.dirname(save_pth), exist_ok=True)
        
        state_dict = self.filter_state_dict(self.state_dict())

        torch.save({"state_dict": state_dict}, save_pth)
        print("Checkpoint saved to", save_pth)

        path = os.path.join(self.cfg.SAVE, f"{self.cfg.DATASETS.TRAIN[0]}.json")
        json.dump(self.task_class_dict, open(path, 'w'))
        print("Task class dict saved to", path)


    def forward(self, images, labels, texts, stage):
        task_prompt = self.extract_image_feat(images)

        text_features = self.extract_text_feat(texts)
        # normlize embeddings
        task_prompt = task_prompt / task_prompt.norm(dim=-1, keepdim=True)         # bs, num_cur_task+1, dim
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)   # num_cls, dim

        # process class vocabulary
        text_embedding, reg_loss = self.process_vocab(text_features) 

        if stage == 1:
            # local cross entropy
            # class token
            cls_logits = self.vlm.logit_scale.exp() * task_prompt[:, -1] @  text_embedding.t()

            # visual token
            if self.prompt_per_task > 0:
                vis_logits = self.vlm.logit_scale.exp() * task_prompt[:, self.num_tasks] @  text_embedding.t()

            if self.prompt_per_task > 0:
                loss = (F.cross_entropy(cls_logits, labels) + F.cross_entropy(vis_logits, labels)) / 2
            else:
                loss = F.cross_entropy(cls_logits, labels)
        elif stage == 2:
            # Negative Class Label Selection
            loss = self.stage_two_forward(task_prompt, text_embedding, labels)

        loss += reg_loss
        return loss
        
    def process_vocab(self, vocab):
        # vocab: bs, dim (after normalize)
        # generate final text embedding
        cur_classes = self.task_class_dict[self.num_tasks]
        if self.cfg.METHOD.MEM:
            # look up vocab
            cur_txt_embed = torch.stack([self.vocab[cls] for cls in cur_classes], dim=0) #num_cls, dim
        else:
            cur_txt_embed = self.init_vocab
        
        cur_txt_embed = cur_txt_embed / cur_txt_embed.norm(dim=-1, keepdim=True)
        text_embedding = (cur_txt_embed + vocab) / 2
        # normalize
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        reg_loss = 0.0
        if self.cfg.METHOD.MEM:
            # stable momentum update
            with torch.no_grad():
                new_embed = self.cfg.METHOD.MOM_COEF * text_embedding.detach().clone() + \
                            (1 - self.cfg.METHOD.MOM_COEF) * cur_txt_embed
                new_embed /= new_embed.norm(dim=-1, keepdim=True)
                self.vocab.update({cls:new_embed[i] for i, cls in enumerate(cur_classes)})
            
            init_vocab = self.init_vocab / self.init_vocab.norm(dim=-1, keepdim=True)
            reg_loss = (text_embedding - init_vocab).norm(dim=-1).mean()
        return text_embedding, reg_loss

    def stage_two_forward(self, task_prompt, text_embedding, labels):
        # previous non-overlapping vocabulary
        non_overlap_cls = list(self.non_overlap_voc_embed.keys())
        non_overlap_voc_embed = torch.stack([self.non_overlap_voc_embed[cls] for cls in non_overlap_cls], dim=0) # num_cls, dim
        non_overlap_voc_embed /= non_overlap_voc_embed.norm(dim=-1, keepdim=True)

        text_embedding = torch.cat([text_embedding, non_overlap_voc_embed], dim=0)  # num_cls + pre_num_cls, dim

        # class token logits
        # bs, num_cls + pre_num_cls
        cls_logits = self.vlm.logit_scale.exp() * task_prompt[:, -1] @  text_embedding.t()

        # visual token logits
        # bs, num_cls + pre_num_cls
        cls_lis =  self.task_class_dict[self.num_tasks] + non_overlap_cls
        if self.prompt_per_task > 0:
            vis_logits = self.get_vis_logits(task_prompt, text_embedding, cls_lis)
        else:
            assert task_prompt.shape[1] == 1
            vis_logits = cls_logits

        # negative energy scores
        with torch.no_grad():
            len_cur_cls = len(self.task_class_dict[self.num_tasks])
            logit_scale = self.vlm.logit_scale.exp()
            cur_neg_energy_score = vis_logits.exp()[:, :len_cur_cls].sum(dim=-1).log() / logit_scale

            # mask unsuitable Negative Class Label
            # 0 for suitable; -inf for unsuitable
            # bs, num_cls + pre_num_cls
            mask = torch.zeros(vis_logits.shape).type_as(vis_logits)

            for task_id in self.task_class_dict:
                if task_id == self.num_tasks:
                    break
                cls_index = [cls_lis.index(cls) for cls in self.task_class_dict[task_id] if cls in non_overlap_cls]

                task_neg_energy_score = vis_logits.exp()[:, cls_index].sum(dim=-1).log() / logit_scale
                gamma = torch.quantile(cur_neg_energy_score - task_neg_energy_score, self.cfg.METHOD.PERCENTAGE)
                # gamma must > 0
                gamma = max(gamma, 0)

                suitable = (cur_neg_energy_score - task_neg_energy_score) > gamma   # bs
                # mask[~suitable][:, cls_index] = float("-inf")
                mask[torch.nonzero(~suitable), torch.LongTensor(cls_index).unsqueeze(0)] = float("-inf")
            
            max_pre_logit = vis_logits.max(dim=-1)[0]
            uncorrect = ~(vis_logits[torch.arange(0, len(vis_logits)), labels] >= max_pre_logit)


        vis_logits += mask
        if self.prompt_per_task > 0:
            cls_logits += mask
        
        # loss
        if self.prompt_per_task > 0:
            # expand local class space 
            loss = (F.cross_entropy(vis_logits[uncorrect], labels[uncorrect], reduction='sum') + \
                    F.cross_entropy(cls_logits[uncorrect], labels[uncorrect], reduction='sum')) / 2
            
            # local class space 
            loss += (F.cross_entropy(vis_logits[~uncorrect][:, :len_cur_cls], labels[~uncorrect], reduction='sum') + \
                    F.cross_entropy(cls_logits[~uncorrect][:, :len_cur_cls], labels[~uncorrect], reduction='sum')) / 2
        else:
             # expand local class space 
            loss = F.cross_entropy(vis_logits[uncorrect], labels[uncorrect], reduction='sum')
            # local class space
            loss += F.cross_entropy(vis_logits[~uncorrect][:, :len_cur_cls], labels[~uncorrect], reduction='sum')
        
        loss /= len(vis_logits)
        return loss
            

    def extract_image_feat(self, images):
        # extract image embedding
        # # bs, num_cur_tasks * prompt_per_task + 1, dim
        devices = list(range(torch.cuda.device_count()))
        if self.prompt_per_task > 0:
            vis_tokens = nn.parallel.data_parallel(self.vlm.visual, images, device_ids=devices, 
                                                   module_kwargs={'task_prompt': True})
        else:
           
            vis_tokens = nn.parallel.data_parallel(self.vlm.visual, images, device_ids=devices)

        if self.prompt_per_task > 0:
            # use task prompt
            task_prompt = vis_tokens[:, :-1].reshape(len(vis_tokens), self.num_tasks+1, self.prompt_per_task, -1).mean(dim=2)  # bs, num_cur_task, dim
            task_prompt = (task_prompt + vis_tokens[:, -1:]) / 2   # v_t
            task_prompt = torch.cat([task_prompt, vis_tokens[:, -1:]], dim=1)   # bs, num_cur_task+1, dim
        else:
            # not use task prompt
            task_prompt = vis_tokens  # bs, 1, dim
        return task_prompt

    def extract_text_feat(self, texts):
        # extract text embedding
        text_features = self.vlm.encode_text(texts)  # num_cls, dim
        return text_features

        
    @torch.no_grad()
    def evaluate(self, images, text_features, cls_mapping):
        self.eval()
        
        # extract image embedding
        task_prompt = self.extract_image_feat(images)

        # get text embedding
        cls_lis = list(text_features.keys())
        txt_embedding = torch.stack([text_features[ele] for ele in cls_lis], dim=0) # num_cls, dim


        # normlize embeddings
        task_prompt = task_prompt / task_prompt.norm(dim=-1, keepdim=True)         # bs, num_cur_task+1, dim
        # fix torch.norm() bug for consistency
        # txt_embedding = txt_embedding / txt_embedding.norm(dim=-1, keepdim=True)   # num_cls, dim
        is_cur = torch.zeros(txt_embedding.shape[0], dtype=torch.bool)
        is_cur[cls_mapping] = True
        txt_embedding[is_cur] /= txt_embedding[is_cur].norm(dim=-1, keepdim=True)
        txt_embedding[~is_cur] /= txt_embedding[~is_cur].norm(dim=-1, keepdim=True)


        if self.prompt_per_task > 0:
            vis_logits = self.get_vis_logits(task_prompt, txt_embedding, cls_lis)
        else:
            logit_scale = self.vlm.logit_scale.exp()
            # fix bug for consistency
            vis_logits = torch.zeros(len(images), txt_embedding.shape[0]).type_as(images)
            vis_logits[:, is_cur]  = logit_scale * task_prompt[:, -1] @ txt_embedding[is_cur].t()
            vis_logits[:, ~is_cur] = logit_scale * task_prompt[:, -1] @ txt_embedding[~is_cur].t()
            # vis_logits = logit_scale * task_prompt[:, -1] @  txt_embedding.t()
        return vis_logits

    def get_vis_logits(self, task_prompt, text_embedding, cls_lis):
        # get logits for visual token
        # task_prompt: bs, num_cur_task+1, dim
        # text_embedding: num_cls + pre_num_cls, dim
        # cls_lis: list of class names

        # bs, num_cls + pre_num_cls
        vis_logits = torch.ones(len(task_prompt), len(cls_lis)).type_as(task_prompt) * float("-inf")
        seen_lis = torch.zeros(len(cls_lis)).bool()

        # per task
        for task_id in self.task_class_dict:
            cls_set = self.task_class_dict[task_id]
            cls_index = [cls_lis.index(cls) for cls in cls_set if cls in cls_lis]

            if len(cls_index) == 0:
                continue

            seen_lis[cls_index] = True
            text_features = text_embedding[cls_index]
            logits = self.vlm.logit_scale.exp() * task_prompt[:, int(task_id)] @  text_features.t()
            
            vis_logits[:, cls_index] = torch.maximum(vis_logits[:, cls_index], logits)
        
        # zero-shot 
        cls_index = [i for i, v in enumerate(seen_lis) if v == False]
        unseen_cls = [cls_lis[i] for i in cls_index]

        if len(unseen_cls) != 0:
            text_features = text_embedding[cls_index]
            logits = self.vlm.logit_scale.exp() * task_prompt[:, -1] @  text_features.t()
            vis_logits[:, cls_index] = torch.maximum(vis_logits[:, cls_index], logits)

        return vis_logits