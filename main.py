import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.func import *
from train import train, evaluate
from utils.metrics import Estimator
from data.builder import generate_dataset
from modules.builder import generate_model
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="./configs", config_name="OCTDL")
def main(cfg: DictConfig) -> None:

    save_path = cfg.base.save_path
    if os.path.exists(save_path):
        if cfg.base.overwrite:
            print_msg('Save path {} exists and will be overwrited.'.format(save_path), warning=True)
        else:
            new_save_path = add_path_suffix(save_path)
            cfg.base.save_path = new_save_path
            warning = 'Save path {} exists. New save path is set to be {}.'.format(save_path, new_save_path)
            print_msg(warning, warning=True)

    os.makedirs(cfg.base.save_path, exist_ok=True)
    worker(cfg)


def worker(cfg):
    if cfg.base.random_seed != -1:
        seed = cfg.base.random_seed
        set_random_seed(seed, cfg.base.cudnn_deterministic)

    log_path = os.path.join(cfg.base.save_path, 'log')
    logger = SummaryWriter(log_path)

    # train
    model = generate_model(cfg)
    train_dataset, test_dataset, val_dataset = generate_dataset(cfg)
    estimator = Estimator(cfg.train.metrics, cfg.data.num_classes, cfg.train.criterion)
    train(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator,
        logger=logger
    )

    # test
    print('Performance of the best validation model:')
    checkpoint = os.path.join(cfg.base.save_path, 'best_validation_weights.pt')
    cfg.train.checkpoint = checkpoint
    model = generate_model(cfg)
    evaluate(cfg, model, test_dataset, estimator)

    print('Performance of the final model:')
    checkpoint = os.path.join(cfg.base.save_path, 'final_weights.pt')
    cfg.train.checkpoint = checkpoint
    model = generate_model(cfg)
    evaluate(cfg, model, test_dataset, estimator)


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


if __name__ == '__main__':
    main()
