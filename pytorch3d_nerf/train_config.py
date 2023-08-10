import sys
sys.path.append("/home/azhuavlev/PycharmProjects/ml-neuman_mano")

from datasets import datamodule
import lighning_models
import lightning as L
from configs import options_manager

def cli_main():
    cli = options_manager.CustomCLI(
        model_class=lighning_models.HandModel,
        datamodule_class=datamodule.MyDatamodule,
        parser_kwargs={"parser_mode": "omegaconf"},
        args=['fit', '--config=/home/azhuavlev/PycharmProjects/ml-neuman_mano/pytorch3d_nerf/configs/config.yaml'],
        save_config_kwargs={"overwrite": True}
    )
    # note: don't call fit!!

if __name__ == "__main__":
    cli_main()