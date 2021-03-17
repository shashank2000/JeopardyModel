import os
import random, torch, numpy
from utils.setup import process_config
import pytorch_lightning as pl
from main import JeopardyModel
from data_module import VQADataModule
import subprocess
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class RealTimeEvalCallback(pl.Callback):
    def __init__(self, checkpoint_dir, downstream_task_config=None, vocab_sz=100, parent_config=None, d2=None):
        self.checkpoint_dir = checkpoint_dir
        self.downstream_task_config = downstream_task_config
        self.vocab_sz = str(vocab_sz)
        self.parent_config = parent_config
        self.downstream2=d2
        self.commands1 = lambda cur_checkpoint: ["python", "test_representation.py", self.downstream_task_config, cur_checkpoint, 
                        self.vocab_sz, self.parent_config, "--gpu-device", "7"]
        self.commands2 = lambda cur_checkpoint: ["python", "test_representation.py", self.downstream2, cur_checkpoint,
                        self.vocab_sz, self.parent_config, "--gpu-device", "7"]    

    def on_fit_end(self, trainer, pl_module):
        cur = trainer.current_epoch
        cur_checkpoint = os.path.join(self.checkpoint_dir, "epoch="+str(cur)+".ckpt")
        commands = self.commands1(cur_checkpoint) + ["-l"]
        subprocess.Popen(commands)

    def on_epoch_end(self, trainer, pl_module):
        cur = trainer.current_epoch
        if cur % 10 == 1:
            cur_checkpoint = os.path.join(self.checkpoint_dir, "epoch="+str(cur)+".ckpt")
            subprocess.Popen(self.commands1(cur_checkpoint))
            if self.downstream2:
                subprocess.Popen(self.commands2(cur_checkpoint))
                
def run(config_path, gpu_device=None):
    config = process_config(config_path)
    seed_everything(config.seed, use_cuda=config.cuda)

    # Perhaps I can change this so that it doesn't save each epoch's results?
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        save_top_k=-1, # could be 5, and that would work fine
        period=1,
    )
    multiple_images = (config.multiple_images)

    wandb_logger = pl.loggers.WandbLogger(name=config.run_name, project=config.exp_name)
    dm = VQADataModule(batch_size=config.optim_params.batch_size, num_workers=config.num_workers, multiple_images=multiple_images)
    num_samples = len(dm.train_dataset)
    mp = config.model_params
    
    model = None
    eval_realtime_callback = None
    my_d2 = config.downstream_task2_config
    if config.system == "jeopardy":
        model = JeopardyModel(config)
    eval_realtime_callback = RealTimeEvalCallback(config.checkpoint_dir, config.downstream_task_config, dm.vl, config_path, d2=my_d2)

    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=[gpu_device],
        max_epochs=config.num_epochs,
        checkpoint_callback=ckpt_callback,
        callbacks=[eval_realtime_callback],
        resume_from_checkpoint=config.continue_from_checkpoint,
        logger=wandb_logger,
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