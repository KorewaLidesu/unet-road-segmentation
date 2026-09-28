from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.dpgb import *
from tools.utils import Lookahead
from tools.utils import process_model_params


# training hparam
max_epoch = 105
ignore_index = len(CLASSES)
train_batch_size = 12
val_batch_size = 12
lr = 1e-3
weight_decay = 0.0025
backbone_lr = 1e-3
backbone_weight_decay = 0.0025
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "RFMUNet"
weights_path = "/root/autodl-tmp/model_weights/dpgb/{}".format(weights_name)
test_weights_name = weights_name
log_name = "/root/autodl-tmp/model_weights/dpgb/{}/lightning_logs/dpgb_log".format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = False

gpus = 'auto'
resume_ckpt_path = None

#  define the network
from geoseg.models.RFMUNet import RFMUNet
net = RFMUNet(num_classes=num_classes)

# define the loss
loss = EdgeLoss(ignore_index=255)
use_aux_loss = False

# define the dataloader
train_dataset = DpgbDataset(data_root=path, mode='train', img_dir='train_images', mask_dir='train_masks', mosaic_ratio=0.25, transform=get_training_transform())
val_dataset = DpgbDataset(data_root=path, mode='val', img_dir='val_images', mask_dir='val_masks', transform=get_validation_transform())
test_dataset = DpgbDataset(data_root=path, mode='test', img_dir='test_images', mask_dir='test_masks', transform=get_validation_transform())

train_loader = DataLoader(dataset=train_dataset,
                        batch_size=train_batch_size,
                        num_workers=4,
                        pin_memory=True,
                        shuffle=True,
                        drop_last=True,
                        persistent_workers=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        persistent_workers=True)

######################## optimizer_config ######################
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
