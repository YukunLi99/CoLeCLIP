
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.prompt_clip import build_vlm
from datasets.name_template import name_temp_info

class PromptZoo(nn.Module):
    def __init__(self, cfg, eval):
        super(PromptZoo, self).__init__()
        self.cfg = cfg
        self.prompt_flag = cfg.METHOD.NAME

        # vlm 
        self.vlm = build_vlm(cfg)
        self.freeze(self.vlm)

        # create prompting module
        emb_d = self.vlm.vis_embed_dim
        tol_num_tasks = len(cfg.TASK_ORDER)
        key_dim = self.vlm.output_dim
        task_id = cfg.TASK_ORDER.index(cfg.DATASETS.TRAIN[0])

        if self.prompt_flag == 'L2P++':
            self.vlm.visual.prompt = L2P(emb_d, tol_num_tasks, task_id, cfg.METHOD.PROMPT_PARAM, key_dim, cfg.METHOD.E_LAYER)
        elif self.prompt_flag == 'DualPrompt':
            self.vlm.visual.prompt = DualPrompt(emb_d, tol_num_tasks, task_id, cfg.METHOD.PROMPT_PARAM, key_dim, cfg.METHOD.E_LAYER, cfg.METHOD.G_LAYER)
        elif self.prompt_flag == 'CODAPrompt':
            self.vlm.visual.prompt = CodaPrompt(emb_d, tol_num_tasks, task_id, cfg.METHOD.PROMPT_PARAM, key_dim, cfg.METHOD.E_LAYER)
        
        # load checkpoints
        self.load_checkpoint()

        # Get the number of learned tasks
        self.num_tasks = cfg.TASK_ORDER.index(cfg.DATASETS.TRAIN[0])

        # CODAPrompt reinit
        if not eval and self.prompt_flag == 'CODAPrompt':
            # reinit the new components for the new task with Gram Schmidt
            if self.num_tasks > 0:
                self.vlm.visual.prompt.reinit()

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
        


    def extract_image_embed(self, images, train):
        devices = list(range(torch.cuda.device_count()))
        with torch.no_grad():
            q, _ = nn.parallel.data_parallel(self.vlm.visual, images, device_ids=devices)

        out, prompt_loss = nn.parallel.data_parallel(self.vlm.visual, images, device_ids=devices, 
                                                     module_kwargs={'prompt':True, 'q':q, 'train': train})

        return out, prompt_loss


    def forward(self, images, labels, texts, stage):
        # loss computation
        assert self.vlm.visual.prompt is not None
        out, prompt_loss = self.extract_image_embed(images, train=True)


        logit = out @ self.classifier_weights + self.classifier_bias # bs, num_cls
        loss = F.cross_entropy(logit, labels) + prompt_loss.sum()
        return loss

    @torch.no_grad()
    def evaluate(self, images, text_features, cls_mapping):
        self.eval()

        weights, bias = text_features

        out, _ = self.extract_image_embed(images, train=False)
        logit = out @ weights + bias # bs, num_cls
        return logit
    
    def freeze(self, model):
        for k, v in model.named_parameters():
            v.requires_grad = False

    def load_checkpoint(self):
        # vlm + prompting module(image)
        # only load prompting module(image)
        if self.cfg.MODEL.LOAD is not None:
            checkpoint = torch.load(self.cfg.MODEL.LOAD)
        else:
            print("not load any checkpoints!")
            return
        
        missing_keys, unexpected_keys = self.load_state_dict(
                checkpoint["state_dict"], strict=False
            )
        # filter vlm parameters
        missing_keys = [k for k in missing_keys if 'vlm' not in k or 'vlm.visual.prompt' in k]
        
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
        
        print("Checkpoint loaded from", self.cfg.MODEL.LOAD)


    # filter parameters which is not saved
    def filter_state_dict(self, state_dict):
        return {
            k: v
            for k, v in state_dict.items()
            if 'vlm.visual.prompt' in k
        }
    
    def save_checkpoint(self, save_pth):
        # save prompting module
        if os.path.dirname(save_pth) != "":
            os.makedirs(os.path.dirname(save_pth), exist_ok=True)
        
        state_dict = self.filter_state_dict(self.state_dict())

        torch.save({"state_dict": state_dict}, save_pth)
        print("Checkpoint saved to", save_pth)


# Implementation From https://github.com/GT-RIPL/CODA-Prompt/blob/main/models/zoo.py
class CodaPrompt(nn.Module):
    # emb_d: dim of value prompt
    # n_tasks: number of total tasks
    # task_count: number of current task
    # prompt_param:  number of prompt components, prompt length, ortho penalty loss weight
    # key_dim: dim of key prompt
    # e_layers: layers which use prompts
    def __init__(self, emb_d, n_tasks, task_count, prompt_param, key_dim=768, e_layers=[0,1,2,3,4]):
        super().__init__()
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self.e_layers = e_layers
        self.task_count = task_count
        self._init_smart(emb_d, prompt_param)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        # self.e_layers = [0,1,2,3,4]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]
        
    def reinit(self):

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        assert task_id == self.task_count

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
            
            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            assert x_querry.device == A.device
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length/2) #prefix tuning
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(nn.Module):
    # emb_d: dim of value prompt
    # n_tasks: number of total tasks
    # task_count: number of current task
    # prompt_param:  # tasks, e-prompt pool length, g-prompt pool length
    # key_dim: dim of key prompt
    # e_layers: layers which use e-prompt
    # g_layers: layers which use g-prompt
    def __init__(self, emb_d, n_tasks, task_count, prompt_param, key_dim=768, e_layers=[2,3,4], g_layers=[0,1]):
        super().__init__()
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self.task_count = task_count
        self._init_smart(emb_d, prompt_param, e_layers, g_layers)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)

    def _init_smart(self, emb_d, prompt_param, e_layers, g_layers):
        
        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = g_layers #[0,1]
        self.e_layers = e_layers #[2,3,4]

        # prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

    # def process_task_count(self):
    #     self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        assert task_id == self.task_count
        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self,f'e_k_{l}') # 0 based indexing here
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            
            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            if train:
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:,task_id]).sum()
                    P_ = p[task_id].expand(len(x_querry),-1,-1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:,k_idx]).sum()
                    P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]
                
            # select prompts
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length/2)
                Ek = P_[:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,i:,:].reshape((B,-1,self.emb_d))
            else:
                i = int(self.e_p_length/2)
                Ek = P_[:,:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,:,i:,:].reshape((B,-1,self.emb_d))
        
        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            Gk = P_[:,:j,:]
            Gv = P_[:,j:,:]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(DualPrompt):
    # emb_d: dim of value prompt
    # n_tasks: number of total tasks
    # task_count: number of current task
    # prompt_param: the size of prompt pool, prompt length, -1 -> shallow, 1 -> deep
    # key_dim: dim of key prompt
    # e_layers: layers which use e-prompt
    def __init__(self, emb_d, n_tasks, task_count, prompt_param, key_dim=768, e_layers=[0,1,2,3,4]):
        super().__init__(emb_d, n_tasks, task_count, prompt_param, key_dim, e_layers, [])

    def _init_smart(self, emb_d, prompt_param, e_layers, g_layers):   
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = e_layers
        else:
            self.e_layers = [0]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p    
