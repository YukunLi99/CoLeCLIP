import os
import torch.nn.functional as F
import copy
import torch
import torch.nn as nn 
from model.clip import clip


# wise_ft
def merge(model_0, model_1, alpha=0.95):
    key_name = [k for k, v in model_0.named_parameters()]
    for i, (param_q, param_k) in enumerate(zip(model_0.parameters(), model_1.parameters())):
        param_k.data = param_k.data * alpha + param_q.data * (1 - alpha)
    return model_1

def merge_we(model_0, model_1, sma_count):
    for param_q, param_k in zip(model_0.parameters(), model_1.parameters()):
        param_k.data = (param_k.data * sma_count + param_q.data) / (1.0 + sma_count)
    return model_1

def wise_we(model_0, model_1, sma_count, model_n, alpha=0.95):
    for param_q, param_k, param_n in zip(model_0.parameters(), model_1.parameters(), model_n.parameters()):
        param_k.data = (
                        (param_k.data * sma_count + param_q.data) / (1.0 + sma_count)
                    ) * alpha + param_n.data * (1-alpha)
    return model_1

def l2_loss(model, model_ref):
    loss = 0.0
    for param_q, param_k in zip(model.parameters(), model_ref.parameters()):
        loss += F.mse_loss(param_q, param_k.detach(), reduction="sum")
    return loss

def distillation(t, s, T=2):
    p = F.softmax(t / T, dim=1)
    loss = F.cross_entropy(s / T, p, reduction="mean") * (T ** 2)
    return loss

def moving_avg(model_0, model_1, alpha=0.999):
    for param_q, param_k in zip(model_0.parameters(), model_1.parameters()):
        param_q.data = param_q.data * alpha + param_k.data * (1 - alpha)

class ZSCL(nn.Module):
    def __init__(self, cfg):
        super(ZSCL, self).__init__()
        self.cfg = cfg
        self.train_mode = cfg.MODEL.TRAIN_MODE
        self.method = cfg.METHOD.NAME
        self.ls = cfg.SOLVER.LS
        self.l2 = cfg.METHOD.L2
        self.T = cfg.METHOD.T
        self.we = cfg.METHOD.WE
        self.ablation_loss_2 = cfg.METHOD.ABLATION_LOSS_2
        self.text_loss = cfg.METHOD.TEXT_LOSS
        self.image_loss = cfg.METHOD.IMAGE_LOSS
        self.weight_adjust = cfg.METHOD.WEIGHT_ADJUST
        self.feature_mse = cfg.METHOD.FEATURE_MSE
        self.moving_avg = cfg.METHOD.MOVING_AVG
        self.we_wise = cfg.METHOD.WE_WISE
        self.avg_freq = cfg.METHOD.AVG_FREQ
        self.we_wise_alpha = cfg.METHOD.WE_WISE_ALPHA
        self.mv_avg_decay = cfg.METHOD.MV_AVG_DECAY
        self.vlm, _, _ = clip.load(cfg.MODEL.VLM.NAME, jit=False)
        # self.vlm = ViT_B_16(cfg.MODEL.VLM.LOAD)
        if cfg.MODEL.LOAD is not None:
            self.load_checkpoint(self.vlm, cfg.MODEL.LOAD)

        if cfg.METHOD.WE_WISE or (cfg.METHOD.WISE_MERGE and cfg.METHOD.WISE_FT_MODEL != "zeroshot"):
            print("Using WiSE-FT with Loaded Model")
            self.model_fix, _, _ = clip.load(cfg.MODEL.VLM.NAME, jit=False)
            # self.model_fix = ViT_B_16(cfg.MODEL.VLM.LOAD)
            self.freeze(self.model_fix)
            if cfg.MODEL.LOAD  is not None:
                self.load_checkpoint(self.model_fix, cfg.MODEL.LOAD)

        if cfg.METHOD.WE or cfg.METHOD.MOVING_AVG or cfg.METHOD.WE_WISE:
            print("Averaging training")
            if cfg.METHOD.MOVING_AVG and cfg.METHOD.MV_AVG_MODEL == "zeroshot": # mv+zeroshot
                self.we_model, _, _ =  clip.load(cfg.MODEL.VLM.NAME, jit=False)
                # self.we_model = ViT_B_16(cfg.MODEL.VLM.LOAD)
                # we_model.cuda()
                self.we_n = 0
            else: #we; mv+m; mv+t; we_wise
                self.we_model = copy.deepcopy(self.vlm)
                # we_model.cuda()
                self.we_n = 0
            self.freeze(self.we_model)
        
        if cfg.METHOD.L2 > 0:
            print("L2 norm")
            self.l2_model = copy.deepcopy(self.vlm)
            self.freeze(self.l2_model)
            # self.l2_model.cuda()

        if cfg.METHOD.NAME == "ZSCL":
        # (Ref Model) get reference model
            print("[Method] ZSCL")
            if cfg.METHOD.REF_MODEL is None:
                if cfg.METHOD.REF_WISE:
                    print("[ref_model] WiSE-Zero-shot")
                    self.ref_model, _, _ = clip.load(cfg.MODEL.VLM.NAME, jit=False)
                    # self.ref_model = ViT_B_16(cfg.MODEL.VLM.LOAD)
                    for param_q, param_k in zip(self.ref_model.parameters(), self.vlm.parameters()):
                        param_q.data = param_q.data * (1 - cfg.METHOD.REF_WISE_ALPHA) + param_k.data * cfg.METHOD.REF_WISE_ALPHA
                else:    
                    print("[ref_model] Zero-shot")
                    self.ref_model, _, _ = clip.load(cfg.MODEL.VLM.NAME, jit=False)
                    # self.ref_model = ViT_B_16(cfg.MODEL.VLM.LOAD)
            else:
                print(f"[ref_model] {cfg.METHOD.REF_MODEL}")
                self.ref_model, _, _ = clip.load(cfg.MODEL.VLM.NAME, jit=False)
                # self.ref_model = ViT_B_16(cfg.MODEL.VLM.LOAD)
                self.load_checkpoint(self.ref_model, cfg.METHOD.REF_MODEL)
            self.ref_model.eval()
            self.freeze(self.ref_model)


    def freeze(self, model):
        for v in model.parameters():
            v.requires_grad = False
        
    def load_checkpoint(self, model, load_pth):
        # if cfg.eval_only and cfg.wise_ft:
        #     print("Use wise-ft.")
        #     model_0 = copy.deepcopy(model)

        checkpoint = torch.load(load_pth)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["state_dict"], strict=False
        )
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
        print("Checkpoint loaded from", load_pth)

        # if cfg.eval_only and cfg.wise_ft:
        #     model = merge(model_0, model, alpha=cfg.alpha)
        # return model


    def forward(self, images, labels, texts, ref_images, ref_labels, ref_texts):
        # -- get text embedding --
        if self.train_mode != "text":
            embeddings = self.vlm(None, texts)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # -- get image embedding --
        out = self.vlm(images, None)
        out = out / out.norm(dim=-1, keepdim=True)


        logits_per_image = self.vlm.module.logit_scale.exp() * out @ embeddings.t()
        loss = F.cross_entropy(logits_per_image, labels, label_smoothing=self.ls)


        if self.l2 > 0:
            loss_l2 = l2_loss(self.vlm, self.l2_model)
            loss += self.l2 * loss_l2

        if self.method == "ZSCL":
            with torch.no_grad():
                # -- get ref text embedding --
                ref_embeddings = self.ref_model(None, ref_texts)
                ref_embeddings = ref_embeddings / ref_embeddings.norm(
                    dim=-1, keepdim=True
                )

                # -- get ref image embedding --
                ref_out = self.ref_model(ref_images, None)
                ref_out = ref_out / ref_out.norm(dim=-1, keepdim=True)

            # -- get image embedding --
            ref_out_current = self.vlm(ref_images, None)
            ref_out_current = ref_out_current / ref_out_current.norm(
                dim=-1, keepdim=True
            )

            # -- loss --
            logits_current = self.vlm.module.logit_scale.exp() * ref_out_current @ ref_embeddings.t()
            logits_ref = self.vlm.module.logit_scale.exp() * ref_out @ ref_embeddings.t()
            loss_ZSCL = distillation(logits_ref, logits_current, T=self.T)

            # feature-space mse
            if self.feature_mse:
                mse_loss = torch.nn.MSELoss()
                loss += mse_loss(ref_out, ref_out_current)

            # -- final loss --
            if self.image_loss:
                if self.weight_adjust:
                    loss = loss + 0.5 * loss_ZSCL 
                else:
                    loss = loss + 1.0 * loss_ZSCL 
            
            # transpose loss
            if self.text_loss:
                logits_current_2 = logits_current.t()
                logits_ref_2 = logits_ref.t()
                loss_ZSCL_2 = distillation(logits_ref_2, logits_current_2, T=self.T)
                if self.weight_adjust:
                    loss += 0.5 * loss_ZSCL_2
                else:
                    loss += loss_ZSCL_2
            
            if self.ablation_loss_2:
                logits_img_current = self.vlm.module.logit_scale.exp() * ref_out_current @ ref_out_current.t()
                logits_img_ref = self.vlm.module.logit_scale.exp() * ref_out @ ref_out.t()
                logits_img_current -= torch.diag(logits_img_current.diag() + 1e4)
                logits_img_ref -= torch.diag(logits_img_ref.diag() + 1e4)
                loss_ZSCL_3 = distillation(logits_img_ref, logits_img_current, T=self.T)
                if self.weight_adjust:
                    loss += 0.5 * loss_ZSCL_3
                else:
                    loss += loss_ZSCL_3
        return loss
    
    def model_we(self, iteration):
        if (self.we or self.moving_avg or self.we_wise) and iteration % self.avg_freq == 0:
            self.we_n += 1
            if self.moving_avg:
                if self.mv_avg_model == "t":
                    next_we_model = copy.deepcopy(self.vlm.module)
                    moving_avg(self.vlm.module, self.we_model, self.mv_avg_decay)
                    self.we_model = next_we_model.cuda()
                else: ### args.moving_avg_model == "n" or "zeroshot"
                    moving_avg(self.vlm.module, self.we_model, self.mv_avg_decay)
            elif self.we:
                merge_we(self.vlm.module, self.we_model, self.we_n)
            else:
                wise_we(self.vlm.module, self.we_model, self.we_n, self.model_fix, self.we_wise_alpha)
    
    def save_checkpoint(self, save_path):
        if self.cfg.METHOD.WISE_MERGE:
            alpha = self.cfg.METHOD.WISE_FT_ALPHA
            if self.cfg.METHOD.WISE_FT_MODEL == "zeroshot":
                wise_ft_model, _, _ =  clip.load(self.cfg.MODEL.VLM.NAME, jit=False)
                # wise_ft_model, _, _ =  ViT_B_16(self.cfg.MODEL.VLM.LOAD)
            else:
                wise_ft_model = copy.deepcopy(self.model_fix)

            wise_ft_model.cuda()
            for param_q, param_k in zip(self.vlm.module.parameters(), wise_ft_model.parameters()):
                param_q.data = param_q.data * alpha + param_k.data * (1 - alpha)
        
        # Saving model
        if self.cfg.SAVE is not None:
            if self.cfg.METHOD.WE or self.cfg.METHOD.WE_WISE:
                to_save_model = self.we_model
            else:
                to_save_model =self.vlm.module

            if os.path.dirname(save_path) != "":
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({"state_dict": to_save_model.state_dict()}, save_path)
            print("Checkpoint saved to", save_path)