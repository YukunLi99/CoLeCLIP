import os
import sys
import argparse
from model.clip import clip
from model.clip.clip import _convert_to_rgb, _transform
import datasets
import pickle
import random

import numpy as np
import torch


def get_datasets_text(ds, args):
    texts = []
    for d in ds:
        ref_sentences_cls = getattr(datasets, d)
        ref_sentences = ref_sentences_cls(
            None,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        ref_template = ref_sentences.template
        ref_texts = [ref_template(x) for x in ref_sentences.classnames]
        texts.extend(ref_texts)
    ret = clip.tokenize(texts).cuda()
    return ret

def virtual_vocab(length=10, n_class=1000):
    voc_len = len(clip._tokenizer.encoder)
    texts = torch.randint(0, voc_len, (n_class, length))
    start = torch.full((n_class, 1), clip._tokenizer.encoder["<start_of_text>"])
    end = torch.full((n_class, 1), clip._tokenizer.encoder["<end_of_text>"])
    zeros = torch.zeros((n_class, 75 - length), dtype=torch.long)

    texts = torch.cat([start, texts, end, zeros], dim=1)
    return texts


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state_dict": classifier.state_dict()}, save_path)
    print("Checkpoint saved to", save_path)

    # with open(save_path, 'wb') as f:
    #     pickle.dump(classifier.cpu(), f)


def torch_load(classifier, save_path, device=None):
    checkpoint = torch.load(save_path)
    # rename the keys in checkpoint
    clip_weight_dict = {'in_proj_weight': 'qkv.weight',
                        'in_proj_bias': 'qkv.bias',
                        'out_proj.weight': 'proj.weight',
                        'out_proj.bias': 'proj.bias'}
    weight_key = list(checkpoint["state_dict"].keys())
    for k in weight_key:
            for clip_k, new_k in clip_weight_dict.items():
                if clip_k in k:
                    checkpoint["state_dict"][k.replace(clip_k, new_k)] = checkpoint["state_dict"].pop(k)
                    continue

    
    missing_keys, unexpected_keys = classifier.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
    print("Checkpoint loaded from", save_path)
    # with open(save_path, 'rb') as f:
    #     classifier = pickle.load(f)

    if device is not None:
        classifier = classifier.to(device)
    return classifier


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge(model_0, model_1, alpha=0.95):
    key_name = [k for k, v in model_0.named_parameters()]
    for i, (param_q, param_k) in enumerate(zip(model_0.parameters(), model_1.parameters())):
        param_k.data = param_k.data * alpha + param_q.data * (1 - alpha)
    return model_1

#from detectron 2
def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    # parser.add_argument(
    #     "--resume",
    #     action="store_true",
    #     help="Whether to attempt to resume from the checkpoint directory. "
    #     "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    # )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--extract-text", action="store_true", help="perform evaluation only")
    parser.add_argument(
            "opts",
            help="""
    Modify config options at the end of the command. For Yacs configs, use
    space-separated "PATH.KEY VALUE" pairs.
    For python-based LazyConfig, use "path.key=value".
            """.strip(),
            default=None,
            nargs=argparse.REMAINDER,
    )
    return parser