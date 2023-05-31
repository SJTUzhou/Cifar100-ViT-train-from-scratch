import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--gpu-id", default="all", help="[all,0,1,...,7]", type=str)
parser.add_argument("--dataset", default="c100", type=str, help="[c10, c100, svhn, imagenet]")
parser.add_argument("--model-name", default="vit", help="[vit]", type=str)
parser.add_argument("--patch", default=8, type=int)
parser.add_argument("--batch-size", default=256, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--max-epochs", default=1000, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=1e-2, type=float) # 5e-5 for adam, 1e-2 for adamw
parser.add_argument("--warmup-epoch", default=10, type=int)
parser.add_argument("--precision", default='bf16-mixed', type=str, help="Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),\
                16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed')")
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--criterion", default="ce")
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--head", default=12, type=int)
parser.add_argument("--num-layers", default=7, type=int)
parser.add_argument("--hidden", default=384, type=int)
parser.add_argument("--mlp-hidden", default=768, type=int)
parser.add_argument("--off-cls-token", action="store_true")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--num-workers", default=8, type=int)
parser.add_argument("--project-name", default="VisionTransformer")
args = parser.parse_args()

###############
### Example ###
###############

# python main.py --autoaugment --label-smoothing --mixup --max-epoch 1000 --num-layers 7 --head 12 --hidden 384 --dropout 0.1 --dataset c100 --patch 8 --mlp-hidden 768 --warmup-epoch 10 --weight-decay 0.01 --seed 42 --batch-size 256 --gpu-id 0

import os
if args.gpu_id != "all":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


import torch
torch.set_float32_matmul_precision("medium")
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import warmup_scheduler
import numpy as np
from utils import get_model, get_dataset, get_experiment_name, get_criterion, save_config_file
from da import CutMix, MixUp


torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.benchmark = True if not args.off_benchmark else False
args.gpus = torch.cuda.device_count()
args.is_cls_token = True if not args.off_cls_token else False




class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        # self.hparams = hparams
        self.hparams.update(vars(hparams))
        self.model = get_model(hparams)
        self.criterion = get_criterion(args)
        if hparams.cutmix:
            self.cutmix = CutMix(hparams.size, beta=1.)
        if hparams.mixup:
            self.mixup = MixUp(alpha=0.8) # [heavy2] in How to train your ViT?
        self.log_image_flag = True
        

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=self.hparams.warmup_epoch, after_scheduler=self.base_scheduler)
        return [self.optimizer], [self.scheduler]
        


    def training_step(self, batch, batch_idx):
        img, label = batch
        if self.hparams.cutmix or self.hparams.mixup:
            if self.hparams.cutmix:
                img, label, rand_label, lambda_= self.cutmix((img, label))
            elif self.hparams.mixup:
                if np.random.rand() <= 0.8:
                    img, label, rand_label, lambda_ = self.mixup((img, label))
                else:
                    img, label, rand_label, lambda_ = img, label, torch.zeros_like(label), 1.
            out = self.model(img)
            loss = self.criterion(out, label)*lambda_ + self.criterion(out, rand_label)*(1.-lambda_)
        else:
            out = self(img)
            loss = self.criterion(out, label)

        if not self.log_image_flag and not self.hparams.dry_run:
            self.log_image_flag = True
            self._log_image(img.clone().detach().cpu())

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("loss", loss)
        self.log("acc", acc)
        return loss


    # pytorch_lightning 2.0.0
    def on_training_epoch_end(self, outputs):
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=self.current_epoch, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", acc, sync_dist=True)
        return loss

    def _log_image(self, image):
        grid = torchvision.utils.make_grid(image, nrow=4)
        self.logger.experiment.log_image(grid.permute(1,2,0))
        print("[INFO] LOG IMAGE!!!")


if __name__ == "__main__":

    train_ds, test_ds = get_dataset(args)
    pin_memory = False if args.dataset=="imagenet" else True
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=pin_memory)

    experiment_name = get_experiment_name(args)
    # Create a TensorBoardLogger to log metrics
    print("[INFO] Log with TensorBoardLogger")
    logger = pl.loggers.TensorBoardLogger('logs/', name=experiment_name)
    refresh_rate = 1
        
    
    net = Net(args)
    

    # pytorch-lightning == 2.0.2
    ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(logger.log_dir,'ckpt'), monitor="val_loss", mode="min", save_top_k=2, 
                                           save_last=True, filename="{epoch}-{val_loss:.4f}-{val_acc:.4f}", every_n_epochs=1)
    
    trainer = pl.Trainer(precision=args.precision, fast_dev_run=args.dry_run, num_nodes=1, devices=args.gpus, 
                        benchmark=args.benchmark, logger=logger, max_epochs=args.max_epochs, callbacks=[ckpt_cb],
                        enable_model_summary=True, enable_progress_bar=True, log_every_n_steps=50, 
                        accelerator="gpu", strategy="ddp", enable_checkpointing=True)
    
    save_config_file(logger.log_dir, args)
    print("log dir: ", logger.log_dir)
    
    trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=test_dl)
    
    if not args.dry_run:
        model_path = f"weights/{experiment_name}.pth"
        torch.save(net.state_dict(), model_path)
