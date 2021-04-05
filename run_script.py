import os
import random, torch, numpy
from utils.setup import process_config
import pytorch_lightning as pl
from main import JeopardyModel
from data_module import VQADataModule
import subprocess
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
                
def run(config_path, gpu_device=None):
    # TODO: add downstream task run functionality
    config = process_config(config_path)
    seed_everything(config.seed, use_cuda=config.cuda)

    # Perhaps I can change this so that it doesn't save each epoch's results?
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        save_top_k=-1, # could be 5, and that would work fine
        period=1,
    )
    multiple_images = (config.multiple_images)
    mp = config.model_params

    wandb_logger = pl.loggers.WandbLogger(name=config.run_name, project=config.exp_name)
    dm = VQADataModule(
        batch_size=config.optim_params.batch_size, q_len=mp.q_len,
        ans_len=mp.ans_len, num_workers=config.num_workers,
        multiple_images=multiple_images, threshold=mp.threshold, 
        mlm_probability=mp.mlm_probability)

    model = None
    if config.system == "jeopardy":
        pretrain_loc = os.environ.get('PRETRAINED_BERT')
        model = JeopardyModel(config, pretrain_loc)


    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=[gpu_device],
        max_epochs=config.num_epochs,
        checkpoint_callback=ckpt_callback,
        resume_from_checkpoint=config.continue_from_checkpoint,
        logger=wandb_logger
    )

    trainer.fit(model, dm)


def seed_everything(seed, use_cuda=True):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda: torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='config/jeopardy_model.json')
    parser.add_argument('--gpu-device', type=int, default=None)
    args = parser.parse_args()
    run(args.config, gpu_device=args.gpu_device)