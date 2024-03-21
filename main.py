import torch
from src import utils
from config.config import get_cfg
from engine.trainer import Trainer


# from detectron2 
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def main(args):
    cfg = setup(args)

    print(cfg)
    utils.seed_all(cfg.SOLVER.SEED)


    if args.eval_only:
        # load model
        model = Trainer.build_model(cfg, eval=True)
        model = model.to(torch.device(cfg.SOLVER.DEVICE))

        if args.extract_text:
            Trainer.extract_ori_txt(model, cfg)
        else:
            Trainer.eval(model, cfg)
    else:
        # set up a trainer
        trainer = Trainer(cfg)
        trainer.train()

if __name__ == "__main__":
    args = utils.default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)


