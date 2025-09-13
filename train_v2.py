import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )

# ---------------- Helper: config schema & grouping ----------------
REQUIRED_KEYS = [
    # dataset / training schedule
    "rec", "num_classes", "num_image", "num_epoch", "warmup_epoch",
    # optimization
    "optimizer", "lr", "weight_decay", "batch_size",
    # arcface / loss
    "margin_list", "embedding_size", "sample_rate",
    # runtime
    "seed", "num_workers", "fp16",
    # logging / verification
    "verbose", "frequent", "val_targets",
]
OPTIONAL_KEYS = [
    "interclass_filtering_threshold", "momentum",
    "dali", "dali_aug", "gradient_acc",
    "resume", "save_all_states", "output",
    # wandb
    "using_wandb", "wandb_key", "wandb_entity", "wandb_project",
    "wandb_log_all", "save_artifacts", "wandb_resume", "suffix_run_name", "notes",
]

BACKBONE_GROUPS = {
    "iresnet_std": {"r18","r34","r50","r100","r200","r2060"},
    "custom_v1": {"r50_custom_v1", "r100_custom_v1"},
    "custom_v2": {"r50_custom_v2", "r100_custom_v2"},
    "fan_ffm": {"r50_fan", "r50_fan_ffm", "r50_ffm"},
    "mobile": {"mbf", "mbf_large"},
    "vit": {"vit_t", "vit_s", "vit_b"},
}

def resolve_backbone_group(name: str) -> str:
    for g, names in BACKBONE_GROUPS.items():
        if name in names:
            return g
    if name.startswith("vit"):
        return "vit"
    return "unknown"

def resolve_output_path(cfg, base_dir="runs"):
    if getattr(cfg, "output", None):
        return cfg.output
    group = resolve_backbone_group(cfg.network)
    ts = datetime.now().strftime("%y%m%d_%H%M")
    # ví dụ: runs/fan_ffm/r50_fan/250913_2130
    return os.path.join(base_dir, group, cfg.network, ts)

def validate_config(cfg):
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    return missing

def print_required_args_help():
    print("\n[train_v2 helper] Required config keys used by train_v2:")
    for k in REQUIRED_KEYS:
        print("  -", k)
    print("\nOptional (commonly used) config keys:")
    for k in OPTIONAL_KEYS:
        print("  -", k)
    print("\nTip: Use `-c/--config` to pass a Python edict config (see --dump-config-template).")

def dump_minimal_template(path=None):
    from textwrap import dedent
    tpl = dedent(f"""\
    from easydict import EasyDict as edict
    import torch
    config = edict()

    # ---- Backbone & loss ----
    config.network = "r50"            # e.g., r50, r50_fan, r50_custom_v2, vit_t
    config.margin_list = (1.0, 0.5, 0.0)
    config.embedding_size = 512

    # ---- Dataset / schedule ----
    config.rec = "faces_webface_112x112"
    config.num_classes = 21083
    config.num_image = 501196
    config.num_epoch = 20
    config.warmup_epoch = 0
    config.seed = 2048
    config.num_workers = 2

    # ---- Optimization ----
    config.optimizer = "sgd"          # or "adamw"
    config.lr = 0.1
    config.weight_decay = 5e-4
    config.batch_size = 128
    config.momentum = 0.9
    config.gradient_acc = 1

    # ---- Runtime ----
    config.fp16 = False
    config.resume = False
    config.save_all_states = False
    config.output = None              # auto-resolve to runs/<group>/<net>/<ts> if None
    config.sample_rate = 1.0
    config.interclass_filtering_threshold = 0

    # ---- Logging / eval ----
    config.verbose = 2000
    config.frequent = 10
    config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

    # ---- DALI (optional) ----
    config.dali = False
    config.dali_aug = False

    # ---- Weights & Biases (optional) ----
    config.using_wandb = False
    config.wandb_key = "XXXX"
    config.wandb_entity = "entity"
    config.wandb_project = "project"
    config.wandb_log_all = True
    config.save_artifacts = False
    config.wandb_resume = False
    config.suffix_run_name = None
    config.notes = ""
    """)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(tpl)
        print(f"[train_v2 helper] Wrote minimal template to: {path}")
    else:
        print(tpl)

def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    
    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        # Sign in to wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity = cfg.wandb_entity, 
                project = cfg.wandb_project, 
                sync_tensorboard = True,
                resume=cfg.wandb_resume,
                name = run_name, 
                notes = cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")
    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, 
        summary_writer=summary_writer, wandb_logger = wandb_logger
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    })
                    
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)
                
        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)
        
        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed ArcFace Training in PyTorch",
        epilog=(
            "Examples:\n"
            "  python train_v2.py -c configs/base.py --print-required\n"
            "  python train_v2.py -c configs/base.py --dump-config-template template_base.py\n"
            "  python train_v2.py -c configs/res50_ffm_onegpu.py --dry-run\n"
            "  python train_v2.py -c configs/res50_ffm_onegpu.py\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to the Python config file (edict named `config`)."
    )
    parser.add_argument(
        "--print-required", action="store_true",
        help="Print required and optional config keys that train_v2 uses and exit."
    )
    parser.add_argument(
        "--dump-config-template", nargs="?", const=True, metavar="PATH",
        help="Print a minimal config template to stdout, or write to PATH if provided."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and show resolved output folder (by backbone group), then exit."
    )

    args = parser.parse_args()

    if args.print_required:
        print_required_args_help()
        raise SystemExit(0)

    if args.dump_config_template is not None:
        # if user provided a path string -> write file, else print to stdout
        if isinstance(args.dump_config_template, str):
            dump_minimal_template(args.dump_config_template)
        else:
            dump_minimal_template(None)
        raise SystemExit(0)

    # load & validate config early (without starting training)
    cfg = get_config(args.config)
    missing = validate_config(cfg)
    if missing:
        print("[train_v2 helper] Missing required config keys:")
        for k in missing:
            print("  -", k)
        raise SystemExit(2)

    # resolve output folder if not provided, based on backbone group
    if not getattr(cfg, "output", None):
        cfg.output = resolve_output_path(cfg)
        print(f"[train_v2 helper] Resolved output to: {cfg.output}")

    if args.dry_run:
        group = resolve_backbone_group(cfg.network)
        print(f"[train_v2 helper] Backbone: {cfg.network}  -> group: {group}")
        print(f"[train_v2 helper] Output dir: {cfg.output}")
        raise SystemExit(0)

    os.makedirs(cfg.output, exist_ok=True)

    main(args)
