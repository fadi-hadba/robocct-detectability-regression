import wandb
import os
import torch.nn as nn
from lightning_network_methods import LightningResNet
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from hydra import compose
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping


@hydra.main(config_path="config", config_name="config_ico_dec_Res")
def main(cfg: DictConfig):
    params = dict(cfg)

    # Erstellen Sie Ihre LightningResNet-Instanz und Ã¼bergeben Sie die Verlustfunktion
    model = LightningResNet(**params)

    # Load the configuration from the YAML file using hydra.compose()
    config = compose(config_name="config_ico_dec_Res")

    wandb.login()

    #{params['model_type']} 

    # TODO some logging routines might have to be adjusted
    wandb_logger = WandbLogger(project="DecIdx_Sphere_iCT", name=f"run_{params['num_epochs']}_{params['base_lr']}_{params['model_type']}")

    # updated
    wandb_logger.watch(model, log="all", log_freq=params['log_every_n_steps'])

    early_stop_callback = EarlyStopping(
        monitor=config.early_stopping.monitor,
        patience=config.early_stopping.patience,
        mode=config.early_stopping.mode,
        min_delta=config.early_stopping.min_delta,
        verbose=config.early_stopping.verbose
    )

    trainer = pl.Trainer(
        max_epochs=params['num_epochs'],
        logger=wandb_logger,
        accelerator='gpu',
        log_every_n_steps=params['log_every_n_steps'],
        check_val_every_n_epoch=params['check_val_every_n_epoch'],
        callbacks=[early_stop_callback],
        fast_dev_run=False
    )

    trainer.fit(model)

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import mean_squared_error, mean_absolute_error
    #
    # # Beispiel Daten generieren
    # np.random.seed(42)
    # true_values = np.random.rand(100)
    # predicted_values = true_values + np.random.normal(0, 0.1, 100)
    #
    # # Berechnen Sie MSE und MAE für verschiedene Epochen
    # epochs = np.arange(1, 101)
    # mse_values = []
    # mae_values = []
    #
    # for epoch in epochs:
    #     mse = mean_squared_error(true_values[:epoch], predicted_values[:epoch])
    #     mae = mean_absolute_error(true_values[:epoch], predicted_values[:epoch])
    #     mse_values.append(mse)
    #     mae_values.append(mae)
    #
    # # Plotten der MSE- und MAE-Kurven
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs, mse_values, label='MSE')
    # plt.plot(epochs, mae_values, label='MAE')
    #
    # plt.title('MSE und MAE Konvergenz')
    # plt.xlabel('Epochen')
    # plt.ylabel('Fehler')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    main()




