import os
import sys
from pathlib import Path

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.lm_dataset_vlm import VLMPretrainDataset, vlm_collate_fn
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    resolve_path,
    setup_seed,
)

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        batch = move_batch_to_device(batch, args.device)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        images = batch["images"] if batch["images"].numel() > 0 else None
        image_grid_thw = batch["image_grid_thw"] if batch["image_grid_thw"].numel() > 0 else None

        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            res = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                images=images,
                image_grid_thw=image_grid_thw,
            )
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, "
                f"aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min"
            )
            if wandb:
                wandb.log(
                    {
                        "loss": current_loss,
                        "logits_loss": current_logits_loss,
                        "aux_loss": current_aux_loss,
                        "learning_rate": current_lr,
                        "epoch_time": eta_min,
                    }
                )

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=args.checkpoint_dir,
            )
            model.train()
            del state_dict

        del batch, input_ids, labels, attention_mask, images, image_grid_thw, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind VLM Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="model save directory")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="resume checkpoint directory")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="tokenizer path")
    parser.add_argument("--vision_tower_path", type=str, default="../weights/rice-vit-large-patch14-560", help="vision tower path")
    parser.add_argument("--image_processor_config", type=str, default="../weights/rice-vit-large-patch14-560/preprocessor_config.json", help="image processor config path")
    parser.add_argument("--image_root", type=str, default="", help="optional image root for json/jsonl datasets")
    parser.add_argument("--save_weight", default="pretrain", type=str, help="saved weight prefix")
    parser.add_argument("--epochs", type=int, default=2, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="initial learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="mixed precision dtype")
    parser.add_argument("--num_workers", type=int, default=8, help="data loader workers")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--log_interval", type=int, default=100, help="log interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="save interval")
    parser.add_argument("--hidden_size", default=768, type=int, help="hidden size")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="number of hidden layers")
    parser.add_argument("--max_seq_len", default=1024, type=int, help="maximum sequence length")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="whether to use MoE")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_i2t.parquet", help="multimodal pretrain dataset")
    parser.add_argument("--from_weight", default="pretrain", type=str, help="base LLM weight name")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="whether to resume")
    parser.add_argument("--freeze_llm", default=1, type=int, choices=[0, 1], help="freeze the language model")
    parser.add_argument("--freeze_vision", default=1, type=int, choices=[0, 1], help="freeze the vision tower")
    parser.add_argument("--freeze_projector", default=0, type=int, choices=[0, 1], help="freeze the projector")
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-VLM-Pretrain", help="wandb project name")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="whether to use torch.compile")
    args = parser.parse_args()

    args.save_dir = str(resolve_path(args.save_dir, SCRIPT_DIR))
    args.checkpoint_dir = str(resolve_path(args.checkpoint_dir, SCRIPT_DIR))
    args.tokenizer_path = str(resolve_path(args.tokenizer_path, SCRIPT_DIR))
    args.vision_tower_path = str(resolve_path(args.vision_tower_path, SCRIPT_DIR))
    args.image_processor_config = str(resolve_path(args.image_processor_config, SCRIPT_DIR))
    args.data_path = str(resolve_path(args.data_path, SCRIPT_DIR))
    args.image_root = str(resolve_path(args.image_root, SCRIPT_DIR)) if args.image_root else None

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.checkpoint_dir)
        if args.from_resume == 1
        else None
    )

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = (
            f"MiniMind-VLM-Pretrain-Epoch-{args.epochs}-"
            f"BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    model, tokenizer = init_model(
        lm_config,
        from_weight=args.from_weight,
        tokenizer_path=args.tokenizer_path,
        save_dir=args.save_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        use_vlm=True,
        vision_tower_path=args.vision_tower_path,
        freeze_llm=bool(args.freeze_llm),
        freeze_vision=bool(args.freeze_vision),
        freeze_projector=bool(args.freeze_projector),
        frozen_module_dtype=(dtype if device_type == "cuda" else None),
    )
    train_ds = VLMPretrainDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        image_root=args.image_root,
        image_processor_config=args.image_processor_config,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check freeze_llm/freeze_vision/freeze_projector.")
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if args.use_compile == 1:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)
        Logger("torch.compile enabled (unsupported subgraphs will fall back to eager)")
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {
            "freqs_cos",
            "freqs_sin",
            "llm.model.freqs_cos",
            "llm.model.freqs_sin",
        }
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=vlm_collate_fn,
        )
        if skip > 0:
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: skip {start_step} steps, "
                f"resume from step {start_step + 1}"
            )
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
