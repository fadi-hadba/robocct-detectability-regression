#import os
#from lightning_network_methods import Detectability_Network
#import hydra
#from omegaconf import DictConfig
#import pytorch_lightning as pl
#from pytorch_lightning.loggers import WandbLogger
#from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor,EarlyStopping


#@hydra.main(version_base=None, config_path="config", config_name="config_ico_dec")
#def main(cfg : DictConfig):
#    params = dict(cfg)
#    model = Detectability_Network(**params)
#    tester = pl.Trainer()
#    trainer.test(model,ckpt_path=params['checkpoint_path'])
    
    

#if __name__ == "__main__":
#    main()

import wandb
import os
import hydra
from hydra import compose
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning_network_methods import LightningResNet


# from lightning_network_methods import Detectability_Network

@hydra.main(version_base=None, config_path="config", config_name="config_ico_dec_Res")
def main(cfg: DictConfig):
    params = dict(cfg)

    # Aktualisieren Sie den checkpoint_path
    params['checkpoint_path'] = r"C:\Users\Abdallah Eid\PycharmProjects\Bachelorarbeit_Fadi\Code_BA\DecIdx_Sphere_iCT\3pjhk579\checkpoints\epoch=199-step=360000.ckpt"

    # Create an instance of your Detectability_Network and pass any necessary parameters
    model = LightningResNet.load_from_checkpoint(**params)

    # Optionally, you can load the configuration from the YAML file using hydra.compose()
    config = compose(config_name="config_ico_dec_Res")
    wandb.login()

    # TODO: Adjust this path based on your project structure

    # Set up the WandbLogger for logging
    wandb_logger = WandbLogger(project="DecIdx_Sphere_iCT_Test",
                               name=f"run_{params['num_epochs']}_{params['base_lr']}_{params['model_type']}")

    # Watch the model and log gradients and model parameters
    wandb_logger.watch(model, log="all")

    # Initialize the Trainer for testing
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator='gpu',
    )

    # Test the model using the specified checkpoint
    trainer.test(model, ckpt_path=params['checkpoint_path'])


if __name__ == "__main__":
    main()

