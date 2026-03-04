import torch
import os
import argparse
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from stedfm.model_builder import get_base_model
from stedfm.DEFAULTS import BASE_PATH
from stedfm.configuration import Configuration
from stedfm.models.dinov2 import DINOv2
from stedfm.modules.transforms import DINOTransform
from stedfm.modules.datamodule import MultiprocessingDataModule
from stedfm.utils import update_cfg


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="STED")
parser.add_argument("--restore-from", type=str, default=None)
parser.add_argument("--model", type=str, default='dinov2-lightning-small')
parser.add_argument("--save-folder", type=str, default=f"{BASE_PATH}/baselines")
parser.add_argument("--dataset-path", type=str, default="./Datasets/FLCDataset/dataset.tar")
### Used only with Hybrid dataset
parser.add_argument("--hpa-path", type=str, default=None)
parser.add_argument("--sim-path", type=str, default=None)
parser.add_argument("--sted-path", type=str, default=None)
###
parser.add_argument("--use-tensorboard", action='store_true')
parser.add_argument("--opts", nargs="+", default=[],
                    help="Additional configuration options")
parser.add_argument("--dry-run", action="store_true",
                    help="Activates dryrun")
args = parser.parse_args()

# Assert args.opts is a multiple of 2
if len(args.opts) == 1:
    args.opts = args.opts[0].split(" ")
assert len(args.opts) % 2 == 0, "opts must be a multiple of 2"

class DataModuleConfig(Configuration):

    num_workers : int = None
    shuffle : bool = True
    use_cache : bool = False
    max_cache_size : float = 32e+9
    return_metadata : bool = False

if __name__=="__main__":

    seed_everything(args.seed, workers=True)
    print(args)
    model, cfg = get_base_model(name=args.model)

    cfg.datamodule = DataModuleConfig()
    cfg.args = args
    update_cfg(cfg, args.opts)

    if args.restore_from:
        OUTPUT_FOLDER = os.path.dirname(args.restore_from)
        print("--- Restoring model ---")
        model = DINOv2.load_from_checkpoint(
            args.restore_from,
            vit=model.student_backbone.vit,
            in_channels=cfg.in_channels,
            n_local_views=cfg.n_local_views,
        )
        print(f"--- Restored model {args.model} from {args.restore_from} successfully ---")
    else:
        OUTPUT_FOLDER = args.save_folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"--- Loaded model {args.model} successfully ---")

    logger = TensorBoardLogger(OUTPUT_FOLDER) if args.use_tensorboard else None

    # DINOv2 uses multi-crop: 2 global (224) + 6 local (96) views
    dinov2_transform = DINOTransform(
        global_crop_size=cfg.global_crop_size,
        local_crop_size=cfg.local_crop_size,
        n_local_views=cfg.n_local_views,
        normalize=None,
    )

    last_model_callback = ModelCheckpoint(
        dirpath=OUTPUT_FOLDER,
        every_n_epochs=1,
        filename="current_model",
        enable_version_counter=False
    )
    last_model_callback.FILE_EXTENSION = ".pth"
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_FOLDER,
        every_n_epochs=100,
        filename="checkpoint-{epoch}",
        save_top_k=-1,
        auto_insert_metric_name=False,
        enable_version_counter=False
    )
    checkpoint_callback.FILE_EXTENSION = ".pth"
    callbacks = [last_model_callback, checkpoint_callback]

    datamodule = MultiprocessingDataModule(args, cfg, transform=dinov2_transform, debug=args.dry_run)

    trainer = Trainer(
        max_epochs=1000,
        devices='auto',
        accelerator='gpu',
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        strategy='ddp_find_unused_parameters_true',
        sync_batchnorm=True,
        use_distributed_sampler=False,
        logger=logger,
        callbacks=callbacks
    )

    trainer.fit(model, train_dataloaders=datamodule, ckpt_path=args.restore_from)
