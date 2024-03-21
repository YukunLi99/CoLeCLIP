import os
import sys
import math
import torch
import copy
import datasets
import pandas as pd
from tqdm import tqdm 
from src import utils, templates
import model.clip.clip as clip

from collections import OrderedDict, Counter
from model.backbone.clip import build_vlm
from datasets.name_template import name_temp_info
from torch.utils.tensorboard import SummaryWriter
from datasets.common import get_dataloader, maybe_dictionarize

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        # model initialization
        self.model = self.build_model(cfg)
        self.wrap_model()

        # data augmentation
        input_resolution = self.model.vlm.module.visual.input_resolution if hasattr(self.model.vlm, 'module') else self.model.vlm.visual.input_resolution
        self.train_augmentation = utils._transform(input_resolution, is_train=True)
        self.test_augmentation = utils._transform(input_resolution, is_train=False)

        # dataset 
        self.train_dataset = self.build_train_dataset(cfg)
        self.test_dataset = self.build_test_dataset(cfg)

        # iteration / epoch
        self.num_batches = len(self.train_dataset.train_loader)
        if cfg.SOLVER.EPOCHS is not None:
            self.epochs = cfg.SOLVER.EPOCHS
            self.tol_iter = cfg.SOLVER.EPOCHS * self.num_batches
        else:
            self.tol_iter = cfg.SOLVER.ITERATIONS
            self.epochs = math.ceil(self.tol_iter / self.num_batches)

        print("Iterations per epoch:", self.num_batches)
        print("Total iterations:", self.tol_iter)


        # optimizer & scheduler
        self.optimizer = self.build_optimizer(cfg)
        self.scheduler = self.build_scheduler(cfg)

        # prepare template
        if cfg.TEMPLATE is not None:
            template = getattr(templates, cfg.template)[0]
        else:
            template = self.train_dataset.template

        # text
        texts = [template(x) for x in self.train_dataset.classnames]
        self.texts = clip.tokenize(texts).cuda()

        # refer dataset & sentences
        if cfg.METHOD.NAME == "ZSCL":
            self.ref_iter, self.ref_texts, self.ref_dataset = self.build_ref_dataset(cfg)
        else:
            self.ref_iter, self.ref_texts, self.ref_dataset = None, None, None
        
        self.ref_dataset_name = cfg.METHOD.REF_DATASET
        self.loss_interval = cfg.SOLVER.LOSS_INTERVAL
        self.tb_writer = SummaryWriter(log_dir=os.path.join(cfg.SAVE, "log", f"{cfg.DATASETS.TRAIN[0]}"))

    @classmethod
    def build_model(cls, cfg, eval=False):
        if cfg.METHOD.NAME == "CLIP":
            model = build_vlm(cfg)
        elif cfg.METHOD.NAME in ['ZSCL', 'Finetune']:
            # need to check
            assert cfg.MODEL.FINETUNE_MODE == 'full' 
            # ZSCL impletation
            if eval:
                model = build_vlm(cfg)
                if cfg.MODEL.LOAD:
                    if cfg.METHOD.WISE_FT:
                        print("Use wise-ft.")
                        model_0 = copy.deepcopy(model)
                    utils.torch_load(model, cfg.MODEL.LOAD)
                    if cfg.METHOD.WISE_FT:
                        model = utils.merge(model_0, model, alpha=cfg.METHOD.ALPHA)
            else:
                from method.ZSCL import ZSCL
                model = ZSCL(cfg)
        
        elif cfg.METHOD.NAME in ['CoLeCLIP']:
            from method.CoLeCLIP import CoLeCLIP
            model = CoLeCLIP(cfg, eval)
            model.zero_shot_vocab = torch.load(cfg.ZERO_SHOT_WEIGHT)
        
        elif cfg.METHOD.NAME in ['LAE']:
            from method.LAE import LAE
            model = LAE(cfg, eval)
        
        elif cfg.METHOD.NAME in ['CODAPrompt', 'L2P++', 'DualPrompt']:
            # need to check
            from method.CODA_Prompt import PromptZoo
            model = PromptZoo(cfg, eval)

        # Apply the same templates for duplicate categories.
        if cfg.METHOD.NAME in ['ZSCL', 'Finetune', 'CLIP']:
            model.dup_cls_template = cls.get_dup_cls_template(cfg)
        elif cfg.METHOD.NAME in ['CoLeCLIP']:
            model.vlm.dup_cls_template = cls.get_dup_cls_template(cfg)
        return model


    def wrap_model(self):
        # move model to device
        self.model = self.model.cuda()
        devices = list(range(torch.cuda.device_count()))
        print("Using devices", devices)
        if self.cfg.METHOD.NAME in ['ZSCL', 'Finetune']:
            self.model.vlm = torch.nn.DataParallel(self.model.vlm, device_ids=devices)
      
        if hasattr(self.model, 'ref_model'):
            self.model.ref_model = torch.nn.DataParallel(self.model.ref_model, device_ids=devices)
    
    def build_train_dataset(self, cfg):
        # prepare dataset
        dataset_class = getattr(datasets, cfg.DATASETS.TRAIN[0])
        dataset = dataset_class(
            self.train_augmentation,
            location=cfg.DATASETS.DATA_LOCATION,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            batch_size_eval=cfg.TEST.BATCH_SIZE,
        )
        return dataset

    def build_test_dataset(self, cfg):
        # prepare dataset
        dataset_class = getattr(datasets, cfg.DATASETS.TRAIN[0])
        dataset = dataset_class(
            self.test_augmentation,
            location=cfg.DATASETS.DATA_LOCATION,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            batch_size_eval=cfg.TEST.BATCH_SIZE,
        )
        return dataset


    def build_ref_dataset(self, cfg):
        # (Ref Dataset) get reference dataset
        ref_dataset_cls = getattr(datasets, cfg.METHOD.REF_DATASET)
        print(f"[Ref Dataset] {cfg.METHOD.REF_DATASET}")
        if cfg.METHOD.REF_DATASET in ["ImageNetSM", "ImageNetSUB"]:
            ref_dataset = ref_dataset_cls(
                self.test_augmentation,
                location=cfg.DATASETS.DATA_LOCATION,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num=cfg.METHOD.NUM,
            )
        else:
            ref_dataset = ref_dataset_cls(
                self.test_augmentation,
                location=cfg.DATASETS.DATA_LOCATION,
                batch_size=cfg.TRAIN.BATCH_SIZE,
            )
        
        ref_iter = iter(ref_dataset.train_loader)

        # (Ref Text) get reference text
        if cfg.METHOD.TEXT_DATASETS  is not None:
            print("[Ref Sentences] Text-Datasets")
            ref_texts = utils.get_datasets_text(cfg.METHOD.TEXT_DATASETS, cfg)
        elif cfg.METHOD.REF_SENTENCES == "random":
            ref_texts = utils.virtual_vocab()
            print("[Ref Sentences] Random Sentences")
        elif cfg.METHOD.REF_SENTENCES is not None:
            ref_sentences_cls = getattr(datasets, cfg.METHOD.REF_SENTENCES)
            print(f"[Ref Sentences] {cfg.METHOD.REF_SENTENCES}")
            ref_sentences = ref_sentences_cls(
                self.test_augmentation,
                location=cfg.DATASETS.DATA_LOCATION,
                batch_size=cfg.TRAIN.BATCH_SIZE,
            )
            if cfg.METHOD.REF_SENTENCES == "conceptual_captions":
                # breakpoint()
                ref_texts = ref_sentences.train_dataset.captions
                ref_texts = clip.tokenize(ref_texts).cuda()

            else:
                ref_template = ref_sentences.template
                ref_texts = [ref_template(x) for x in ref_sentences.classnames]
                ref_texts = clip.tokenize(ref_texts).cuda()
        else:
            print(f"[Ref Sentences] {cfg.METHOD.REF_DATASET}")
            ref_template = ref_dataset.template
            ref_texts = [ref_template(x) for x in ref_dataset.classnames]
            ref_texts = clip.tokenize(ref_texts).cuda()

        return ref_iter, ref_texts, ref_dataset
    
    def build_optimizer(self, cfg):
        if cfg.MODEL.FINETUNE_MODE == 'full':
            if cfg.MODEL.TRAIN_MODE == "text":
                print("[Training mode] Text Encoder")
                visual_params_name = [k for k, v in self.model.vlm.module.visual.named_parameters()]
                exclude_params_name = visual_params_name + ["logit_scale"]
                params = [
                    v for k, v in self.model.vlm.module.named_parameters() if k not in exclude_params_name
                ]
            elif cfg.MODEL.TRAIN_MODE == "image":
                print("[Training mode] Image Encoder")
                params = self.model.vlm.module.visual.parameters()
            else:
                assert cfg.MODEL.TRAIN_MODE == "whole"
                print("[Training mode] Both Encoders")
                exclude_params_name = ["logit_scale"]
                params = [
                    v for k, v in self.model.vlm.module.named_parameters() if k not in exclude_params_name
                ]

            # optimizer
            optimizer = torch.optim.AdamW(
                params, lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WD, betas=(0.9, cfg.SOLVER.BETA2)
            )
        elif cfg.MODEL.FINETUNE_MODE == 'peft':
            params = [
                    v for k, v in self.model.named_parameters() if v.requires_grad
                ]
            
            # optimizer
            optimizer = torch.optim.Adam(params, lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WD)
        return optimizer

    @classmethod
    def get_dup_cls_template(cls, cfg):
        # total class set
        classset = []
        for d in cfg.TASK_ORDER:
            classset.extend(name_temp_info[d][0])
        
        # get duplicate classes
        count = Counter(classset)
        dup_cls = [ele for ele in count if count[ele]> 1]
        dup_cls_template = {}

        # merge templates for all categories except forest and river, 
        # using the template from EuroSAT
        for cls in dup_cls:
            for d in cfg.TASK_ORDER:
                if cls in name_temp_info[d][0]:
                    if cls not in ['forest', 'river']:
                        if cls not in dup_cls_template:
                            dup_cls_template[cls] = []
                        dup_cls_template[cls].extend(name_temp_info[d][1])
                    elif d == 'EuroSAT':
                        dup_cls_template[cls] = name_temp_info[d][1]
        return dup_cls_template


    def build_scheduler(self, cfg):
        if cfg.MODEL.FINETUNE_MODE == 'full':
            scheduler = utils.cosine_lr(
            self.optimizer, cfg.SOLVER.LR, cfg.SOLVER.WARMUP_LENGTH, self.tol_iter
        )
        elif cfg.MODEL.FINETUNE_MODE == 'peft':
            #  Learning rate unchanged.
            if self.cfg.SOLVER.SCHEDULE_UNIT == "none":
                num_steps = float("inf")
            T = cfg.SOLVER.CYCLE if cfg.SOLVER.CYCLE > 0 else num_steps
            s_min = cfg.SOLVER.SCALE_MIN
            scale = lambda r: r * (1 - s_min) + s_min
            lr_lambda = lambda t: scale(1 - t / T)
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        return scheduler


    @classmethod
    def get_cls_target(cls, cfg):
        # get classset and target for CIL
        classset = []
        targets = []

        # Get the number of learned tasks
        if cfg.TEST.CUR_TASK is not None:
            num_tasks = cfg.TASK_ORDER.index(cfg.TEST.CUR_TASK)
        else:
            num_tasks = cfg.TASK_ORDER.index(cfg.DATASETS.TRAIN[0])

        for d in cfg.TASK_ORDER[:num_tasks+1]:
            classset.extend(name_temp_info[d][0])

        # Deduplication classes
        dedup_classset = list(OrderedDict.fromkeys(classset))

        # target for each seen dataset
        for d in cfg.TASK_ORDER[:num_tasks+1]:
            cls_name = name_temp_info[d][0]
            targets.append([dedup_classset.index(cls) for cls in cls_name])
        
        return  dedup_classset, targets

        
    def preprocess_vocab(self):
        # find non-overlapping categories
        if self.cfg.DATASETS.TRAIN[0] == self.cfg.TASK_ORDER[0]:
            # first task
            self.model.dup_cls = []
        else:
            pre_cls = []
            for d in self.cfg.TASK_ORDER:
                if d == self.cfg.DATASETS.TRAIN[0]:
                    break
                else:
                    pre_cls.extend(name_temp_info[d][0])

            # Deduplication classes
            pre_cls = list(OrderedDict.fromkeys(pre_cls))

            assert name_temp_info[self.cfg.DATASETS.TRAIN[0]][0] == self.train_dataset.classnames

            # Duplicate categories with current task
            dup_cls = list(set(pre_cls) & set(name_temp_info[self.cfg.DATASETS.TRAIN[0]][0]))
            self.model.dup_cls = dup_cls

            if self.cfg.METHOD.NAME in ['CoLeCLIP']:
                self.model.non_overlap_voc_embed = {cls:self.model.vocab[cls] for cls in pre_cls if cls not in dup_cls}
                
            
        # update vocabulary
        cur_vocab = {cls: self.model.zero_shot_vocab[cls] for cls in name_temp_info[self.cfg.DATASETS.TRAIN[0]][0] if cls not in self.model.dup_cls}
        self.model.vocab.update(cur_vocab)
        self.model.init_vocab = torch.stack([self.model.vocab[cls] for cls in name_temp_info[self.cfg.DATASETS.TRAIN[0]][0]], dim=0)

        # store classes for the current task
        task_id = self.cfg.TASK_ORDER.index(self.cfg.DATASETS.TRAIN[0])
        self.model.task_class_dict[task_id] = name_temp_info[self.cfg.DATASETS.TRAIN[0]][0]
            

    def train(self):
        assert name_temp_info[self.cfg.DATASETS.TRAIN[0]][0] == self.train_dataset.classnames
        if self.cfg.METHOD.USE_VOC:
            self.preprocess_vocab()
        
        for epoch in range(self.epochs):
            if self.cfg.SOLVER.SCHEDULE_UNIT == 'epoch' and epoch > 0:
                self.scheduler.step()
            
            mean_loss = self.train_one_loop(data_loader=self.train_dataset.train_loader, epoch=epoch)

            # evaluation for each epoch
            if self.cfg.SOLVER.EVAL_EVERY_EPOCH:
                self.eval_epoch(epoch)
            
            tags = ["loss", "learning_rate"]
            self.tb_writer.add_scalar(tags[0], mean_loss, epoch)
            self.tb_writer.add_scalar(tags[1], self.optimizer.param_groups[0]["lr"], epoch)

        # save
        self.save_model()
        

    def save_model(self):
        # save VLM
        path = os.path.join(self.cfg.SAVE, f"{self.cfg.DATASETS.TRAIN[0]}.pth")
        self.model.save_checkpoint(path)

        # update vocab / classifier for current task
        if self.cfg.METHOD.NAME in ['LAE', 'CODAPrompt', 'L2P++', 'DualPrompt']: 
            # update classifier
            self.model.classifier.update({cls: (self.model.classifier_weights[:, i].detach(), self.model.classifier_bias[:, i].detach()) for i, cls in enumerate(name_temp_info[self.cfg.DATASETS.TRAIN[0]][0])})
            extra_path = os.path.join(self.cfg.SAVE, f"{self.cfg.DATASETS.TRAIN[0]}_classifier.pth")

        elif self.cfg.METHOD.NAME in ['CoLeCLIP']:
            if not self.cfg.METHOD.MEM:
                # baseline without stable momentum update
                cur_vocab = self.model.zeroshot_classifier([name_temp_info[self.cfg.DATASETS.TRAIN[0]]])
                # fusion from cur and pre
                assert list(cur_vocab.keys()) == name_temp_info[self.cfg.DATASETS.TRAIN[0]][0]
                for i, c in enumerate(cur_vocab):
                    txt_embed = (cur_vocab[c] / cur_vocab[c].norm() + self.model.init_vocab[i] / self.model.init_vocab[i].norm()) / 2
                    self.model.vocab[c] = txt_embed / txt_embed.norm()
            extra_path = os.path.join(self.cfg.SAVE, f"{self.cfg.DATASETS.TRAIN[0]}_vocab.pth")

        else:
            extra_path = None

        # save vocab / classifier
        if extra_path is not None:
            if hasattr(self.model, 'classifier'):
                torch.save(self.model.classifier, extra_path)
            else:
                torch.save(self.model.vocab, extra_path)
            print("vocab / classifier saved to", extra_path)


    def eval_epoch(self, epoch):
        self.model.eval()
        if self.cfg.MODEL.FINETUNE_MODE == 'full':
            # zscl
            if self.cfg.METHOD.WE or self.cfg.METHOD.WE_WISE:
                model = self.model.we_model
            else:
                model = self.model.vlm.module
        else:
            model = self.model
        
        if self.cfg.SOLVER.EVAL_EVERY_EPOCH:
            # update vocab / classifier for current task
            if self.cfg.METHOD.NAME in ['LAE', 'CODAPrompt', 'L2P++', 'DualPrompt']: 
                # update classifier
                model.classifier.update({cls: (model.classifier_weights[:, i].detach(), model.classifier_bias[:, i].detach()) for i, cls in enumerate(name_temp_info[self.cfg.DATASETS.TRAIN[0]][0])})
            elif self.cfg.METHOD.NAME in ['CoLeCLIP'] and not self.cfg.METHOD.MEM:
                # baseline without stable momentum update
                cur_vocab = model.zeroshot_classifier([name_temp_info[self.cfg.DATASETS.TRAIN[0]]])
                # fusion from cur and pre
                assert list(cur_vocab.keys()) == name_temp_info[self.cfg.DATASETS.TRAIN[0]][0]
                for i, c in enumerate(cur_vocab):
                    txt_embed = (cur_vocab[c] / cur_vocab[c].norm() + model.init_vocab[i] / model.init_vocab[i].norm()) / 2
                    model.vocab[c] = txt_embed / txt_embed.norm()
            elif self.cfg.METHOD.NAME in ['ZSCL', 'Finetune']:
                # extract text embedding
                model.vocab = model.zeroshot_classifier(name_temp_info[self.cfg.DATASETS.TRAIN[0]])
            
            # get classes for CIL  and convert target
            classset, targets = self.get_cls_target(self.cfg)

            til_top1, cil_top1 = self.eval_single_dataset(model, self.test_dataset, self.cfg, classset, targets[-1])
        
            print("[epoch {}] TIL accuracy: {}".format(epoch, round(til_top1, 3)))
            print("[epoch {}] CIL accuracy: {}".format(epoch, round(cil_top1, 3)))

            tags = ["TIL_ACC", "CIL_ACC"]
            self.tb_writer.add_scalar(tags[0], til_top1, epoch)
            self.tb_writer.add_scalar(tags[1], cil_top1, epoch)



    def train_one_loop(self, data_loader, epoch):
        # train mode
        if self.cfg.MODEL.FINETUNE_MODE == 'full':
            self.model.vlm.train()
        elif self.cfg.MODEL.FINETUNE_MODE == 'peft':
            self.model.train()
        
        data_loader = tqdm(data_loader, file=sys.stdout)
        mean_loss = torch.zeros(1).cuda()

        for step, data in enumerate(data_loader):
            if self.cfg.SOLVER.SCHEDULE_UNIT == 'step':
                self.scheduler(step + epoch * self.num_batches)
            
            # LAE freeze peft for the first 3/5 iterations.
            # need to check tommorow
            if self.cfg.METHOD.NAME in ['LAE'] and self.cfg.TASK_ORDER.index(self.cfg.DATASETS.TRAIN[0]) != 0:
                if (step + epoch * self.num_batches) < (self.tol_iter * 3 / 5):
                    if next(self.model.vis_pets.parameters()).requires_grad:
                        print("===> Freeze pets")
                        self.model.freeze_pets()
                else:
                    if not next(self.model.vis_pets.parameters()).requires_grad:
                        print("===> Unfreeze pets")
                        self.model.unfreeze_pets()

            images, labels = data
            images, labels = images.cuda(), labels.cuda()

            # zscl
            if self.ref_iter is not None:
                # FIXME: only for ImageNet
                if self.ref_dataset_name in ["ImageNet", "ImageNetSM", "ImageNetSUB"]:
                    try:
                        ref_batch = next(self.ref_iter)
                    except:
                        self.ref_iter = iter(self.ref_dataset.train_loader)
                        ref_batch = next(self.ref_iter)
                    ref_images, ref_labels = ref_batch["images"], ref_batch["labels"]
                else:
                    try:
                        ref_images, ref_labels = next(self.ref_iter)
                    except:
                        self.ref_iter = iter(self.ref_dataset.train_loader)
                        ref_images, ref_labels = next(self.ref_iter)
                ref_images, ref_labels = ref_images.cuda(), ref_labels.cuda()
            else:
                ref_images, ref_labels =  None, None

            # forward
            if self.cfg.MODEL.FINETUNE_MODE == 'full':
                loss = self.model(images, labels, self.texts, ref_images, ref_labels, self.ref_texts)
            elif self.cfg.MODEL.FINETUNE_MODE == 'peft':
                stage = 1
                is_stage_two = self.cfg.METHOD.STAGE_STEP is not None and (step + epoch * self.num_batches) > int(self.cfg.METHOD.STAGE_STEP * self.tol_iter)
                num_tasks = self.cfg.TASK_ORDER.index(self.cfg.DATASETS.TRAIN[0])
                if num_tasks > 0 and self.cfg.METHOD.STAGE_STEP is not None and  is_stage_two:
                    stage = 2
                loss = self.model(images, labels, self.texts, stage)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log for loss
            if (step + epoch * self.num_batches) % self.loss_interval == 0:
                tags = ["loss_iter", "learning_rate_iter"]
                self.tb_writer.add_scalar(tags[0], loss.item(), step + epoch * self.num_batches)
                self.tb_writer.add_scalar(tags[1], self.optimizer.param_groups[0]["lr"], step + epoch * self.num_batches)
            
            # zscl
            if hasattr(self.model, 'model_we'):
                self.model.model_we(step + epoch * self.num_batches)
            
            # LAE ema update pet
            if hasattr(self.model, 'post_train_step'):
                self.model.post_train_step()
            
            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)
            
            if (step + epoch * self.num_batches) == self.tol_iter:
                break
        return mean_loss.item()


    @classmethod
    def eval(cls, model, cfg):

        assert cfg.DATASETS.EVAL != None, 'Error: Eval datasets is None!'
        assert cfg.TEST.CUR_TASK != None, 'Error: Not provided with the name of the current task that model has learned'

        if len(cfg.DATASETS.EVAL) == len(cfg.TASK_ORDER):
            # inference mode: inference for all datasets

            # save_dir
            til_save_dir = os.path.join(cfg.SAVE, 'TIL_results.csv')
            cil_save_dir = os.path.join(cfg.SAVE, 'CIL_results.csv')

            til_eval_acc = {}
            cil_eval_acc = {}
           
            # Get the number of learned tasks
            num_tasks = cfg.TASK_ORDER.index(cfg.TEST.CUR_TASK)

            assert cfg.DATASETS.EVAL == cfg.TASK_ORDER, 'Ensure that the order of evaluation datasets matches the order of tasks.'

            # get classes for CIL  and convert target
            classset, targets = cls.get_cls_target(cfg)


            for i, dataset_name in enumerate(cfg.DATASETS.EVAL):
                if cfg.METHOD.NAME in ['LAE', 'CODAPrompt', 'L2P++', 'DualPrompt']:
                    if i > num_tasks:
                        # can not eval zero-shot performance
                        break 
                
                # datasets
                print("Evaluating on", dataset_name)
                dataset_class = getattr(datasets, dataset_name)
                input_resolution = model.vlm.visual.input_resolution if hasattr(model, 'vlm') else model.visual.input_resolution
                dataset = dataset_class(
                    utils._transform(input_resolution, is_train=False),
                    location=cfg.DATASETS.DATA_LOCATION,
                    batch_size=cfg.TRAIN.BATCH_SIZE,
                    batch_size_eval=cfg.TEST.BATCH_SIZE,
                )
            
                assert dataset.classnames == name_temp_info[dataset_name][0], 'The order of classes needs to be maintained.'
                cls_name = name_temp_info[dataset_name][0] if i > num_tasks else classset
                target = list(range(len(cls_name))) if i > num_tasks else targets[i]

                til_top1, cil_top1 = cls.eval_single_dataset(model, dataset, cfg, cls_name, target)
                til_eval_acc[dataset_name] = [til_top1]
                cil_eval_acc[dataset_name] = [cil_top1]


            # save evaluation results
            cls.save_res(til_eval_acc, til_save_dir, cfg.TEST.CUR_TASK)
            cls.save_res(cil_eval_acc, cil_save_dir, cfg.TEST.CUR_TASK)



    @classmethod
    def save_res(cls, eval_acc, save_dir, cur_task):
        res = pd.DataFrame(eval_acc)
        res.insert(0, 'stage', cur_task)
        if os.path.exists(save_dir):
            pre_res = pd.read_csv(save_dir)
            res = pd.concat([pre_res, res])
        if os.path.dirname(save_dir) != "":
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        res.to_csv(save_dir,  index=False)


    @classmethod
    def get_text_embed(cls, cfg, cls_name, model):
        if cfg.METHOD.NAME in ['ZSCL', 'Finetune', 'CLIP']:
            assert cfg.METHOD.USE_VOC is False
            # extract text embeddings in advance
            cls_temp = [name_temp_info[t] for t in cfg.TASK_ORDER]

            model.vocab = model.zeroshot_classifier(cls_temp)
        
        text_features = []

        if cfg.METHOD.NAME in ['LAE', 'CODAPrompt', 'L2P++', 'DualPrompt']:
            weights = []
            bias = []
            for cls in cls_name:
                weights.append(model.classifier[cls][0])
                bias.append(model.classifier[cls][1])
            
            # weights: dim, num_cls
            # bias: 1, num_cls
            text_features = (torch.stack(weights, dim=-1), torch.stack(bias, dim=-1))
        else:
            for cls in cls_name:
                if cls in model.vocab:
                    # in-vocab
                    text_features.append(model.vocab[cls])
                else:
                    # not-in-vocab
                    # load from zero_shot_vocab
                    text_features.append(model.zero_shot_vocab[cls])
            
            if cfg.METHOD.NAME == 'CoLeCLIP':
                text_features = {cls_name[i]:ele for i, ele in enumerate(text_features)}
            else:
                text_features = torch.stack(text_features, dim=1)  # dim, num_cls
        
        return text_features
        
        
    
    @classmethod
    def eval_single_dataset(cls, model, dataset, cfg, cls_name, target):
        model.eval()

        
        # get text embeddings for inference
        text_features = cls.get_text_embed(cfg, cls_name, model)
    
        # get dataloader
        dataloader = get_dataloader(
            dataset, is_train=False, args=cfg, image_encoder=None
        )

        # evaluate
        if isinstance(text_features, dict):
            assert list(text_features.keys()) == cls_name

        til_top1, til_top5, cil_top1, cil_top5 = cls.zeroshot_eval(model, dataloader, text_features, target)

        print(f"Top-1 accuracy for TIL: {til_top1:.2f}")
        if len(cls_name) != len(target):
            print(f"Top-1 accuracy for CIL: {cil_top1:.2f}")
        return til_top1, cil_top1



    @classmethod
    @torch.no_grad()
    def zeroshot_eval(cls, model, loader, text_features, cls_mapping):
        til_top1, til_top5, cil_top1, cil_top5, n = 0.0, 0.0, 0.0, 0.0, 0.0
        for i, data in enumerate(tqdm(loader)):

            data = maybe_dictionarize(data)
            images = data["images"].cuda()
            target = data["labels"].cuda()

            # predict
            # fix bug
            if torch.is_tensor(text_features):
                logits = model.evaluate(images, text_features.clone(), cls_mapping)
            else:
                logits = model.evaluate(images, text_features, cls_mapping)

            # measure accuracy
            # 1) convert target
            new_target = torch.tensor(cls_mapping).type_as(target)[target]

            # 2) cil / til eval
            if isinstance(logits, tuple):
                # LAE (cil_logit, til_logit)
                til_acc1, til_acc5 = cls.accuracy(logits[1], target, topk=(1, 5))
                cil_acc1, cil_acc5 = cls.accuracy(logits[0], new_target, topk=(1, 5))
            else:
                til_acc1, til_acc5 = cls.accuracy(logits[:, cls_mapping], target, topk=(1, 5))
                cil_acc1, cil_acc5 = cls.accuracy(logits, new_target, topk=(1, 5))

            til_top1 += til_acc1
            til_top5 += til_acc5

            cil_top1 += cil_acc1
            cil_top5 += cil_acc5
            n += images.size(0)

        til_top1 = (til_top1 / n) * 100
        til_top5 = (til_top5 / n) * 100

        cil_top1 = (cil_top1 / n) * 100
        cil_top5 = (cil_top5 / n) * 100

        return til_top1, til_top5, cil_top1, cil_top5


    @classmethod
    def accuracy(cls, output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [
            float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            for k in topk
        ]

    @classmethod
    def extract_ori_txt(cls, model, cfg):
        assert cfg.METHOD.NAME == "CLIP"
        cls_temp = [name_temp_info[t] for t in cfg.TASK_ORDER]
        text_features = model.zeroshot_classifier(cls_temp)
        torch.save(text_features, cfg.ZERO_SHOT_WEIGHT)
