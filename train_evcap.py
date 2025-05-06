import os
import torch
import itertools
from torch.utils.data import DataLoader, DistributedSampler
import random
import sys
import argparse
import numpy as np
import utils
from optims import LinearWarmupCosineLRScheduler, set_optimizer
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize
import torchvision.transforms as transforms
from dataset.coco_dataset import COCODataset
from models.evcap import EVCap
from common.dist_utils import (
    get_rank,
    init_distributed_mode,
    get_world_size,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model,optimizer, cur_epoch, cur_iter, output_dir):
    """
    Save the checkpoint at the current epoch.
    """
    model_no_ddp = model
    param_grad_dic = {
        k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
    }
    state_dict = model_no_ddp.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": cur_epoch,
        "iter": cur_iter,
    }
    save_path = os.path.join(output_dir, f"checkpoint_{cur_epoch}_{cur_iter}.pt")
    print(f"Saving checkpoint at epoch {cur_epoch}, iter {cur_iter} to {save_path}.")
    torch.save(save_obj, save_path)

def load_checkpoint(model, optimizer, checkpoint_path=None):
    """
    Load the checkpoint if available.
    """
    checkpoint_path = checkpoint_path or '/kaggle/input/checkpoint/weights.pt'
    print(f"Loading checkpoint from {checkpoint_path}.")
    try:
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
            return model, optimizer, 0, 0  # Start from scratch

        checkpoint = torch.load(checkpoint_path)
        for k, v in checkpoint['model'].items():
            print(k, v.shape)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        # if missing_keys:
        #     print(f"Warning: Missing keys in state_dict: {missing_keys}")
        # if unexpected_keys:
        #     print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")

        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0)
        start_iter = checkpoint.get("iter", 0)
        print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}, iter {start_iter}.")
        return model, optimizer, start_epoch, start_iter

    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        return model, optimizer, 0, 0  # Start from scratch


def train(dataset, model, args):
    device = torch.device(f"cuda:{get_rank()}")
    batch_size = args.bs
    epochs = args.epochs
    accum_grad_iters = 1
    output_dir = args.out_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            num_replicas=get_world_size(),
            rank=get_rank(),
        )
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[get_rank()])
    else: 
        sampler = None
        model = model.to(device)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, sampler=sampler,shuffle=False, drop_last=True)
    
    
    model.train()
    optimizer = set_optimizer(model, init_lr=1e-4, weight_decay=0.05)
    scheduler = LinearWarmupCosineLRScheduler(optimizer= optimizer,
                max_epoch=epochs,
                iters_per_epoch=len(train_dataloader),
                min_lr=8e-5,
                init_lr=1e-4,
                decay_rate=None,
                warmup_start_lr=1e-6,
                warmup_steps=5000,)
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    use_amp = scaler is not None
    print('use_amp', use_amp)

    model, optimizer, start_epoch, start_iter = load_checkpoint(model, optimizer)
    global_iter = start_iter
    print("gloab iter: ", global_iter)

    train_dataloader_skipped = itertools.islice(train_dataloader, global_iter, None)

    for epoch in range(start_epoch, epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.update(loss=1000.0)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        print_freq = 50
        header = 'Train Epoch: [{}]'.format(epoch)
        for idx, samples in enumerate(metric_logger.log_every(train_dataloader_skipped, len(train_dataloader), global_iter, print_freq, header), start=global_iter):
            print(f"Iter {idx + 1}/{len(train_dataloader)}: Processing batch...")
            samples['image'] = samples['image'].to(device)

            # print("samples['image'] ----> ", samples['image'][0][0][0])
            
            scheduler.step(cur_epoch=epoch, cur_step=idx)    
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model(samples)["loss"]
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (idx + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr = optimizer.param_groups[0]["lr"])

            if (idx + 1) % 50 == 0:
                save_checkpoint(model, optimizer, epoch, idx + 1, output_dir)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger.global_avg())
 
        if epoch == epochs - 1:
            # output_dir_model = os.path.join(output_dir, f"{epoch:03d}.pt")
            save_checkpoint(model, optimizer, epoch, len(train_dataloader), output_dir)
    return model


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print('Starts ...')
    print(" # PID :", os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='/kaggle/working/checkpoints')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--bs', type=int, default=6)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--device', default = 'cuda', help = 'gpu for training')
    parser.add_argument('--distributed', default = False)
    parser.add_argument('--amp', default = True)
    parser.add_argument('--dist_url', default = "env://")
    parser.add_argument('--world_size', type = int, default = 1)
    parser.add_argument('--num_query_token_txt', type = int, default = 8)
    parser.add_argument('--topn', type = int, default = 9)
    parser.add_argument('--disable_random_seed', action = 'store_true', default = False, help = 'set random seed for reproducing')
    parser.add_argument('--random_seed', type = int, default = 42, help = 'set random seed for reproducing')
    args = parser.parse_args()
    print(f'args: {vars(args)}')
    if not args.disable_random_seed:
        set_seed(args.random_seed)
    init_distributed_mode(args)
    print(f'args: {vars(args)}')
    data_root = '/kaggle/input/ms-coco-dataset'
    dataset = COCODataset(data_root=data_root) #x=16 for testing on 16 images
    model_type = "lmsys/vicuna-7b-v1.3"
    model = EVCap(
            ext_path = '/kaggle/input/mydataset/EVCap-main/ext_data/ext_memory_lvis.pkl',
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            freeze_qformer=True,
            num_query_token=32,
            num_query_token_txt=args.num_query_token_txt,
            topn = args.topn,
            llama_model=model_type,
            prompt_path="/kaggle/input/mydataset/EVCap-main/prompts/prompt_evcap.txt",
            prompt_template='###Human: {} ###Assistant: ',
            max_txt_len=128,
            end_sym='\n',
            low_resource=True,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    )
    train(dataset, model, args)


if __name__ == '__main__':
    main()
