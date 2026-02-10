import os
import torch
import numpy as np
from torch import nn
from torchvision import transforms
from torch.nn import functional as F
from torch.optim import SGD, Adam, RMSprop, Adagrad, Adadelta, AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from dataloader_proj import Detectability_Projections_Dataset
from Network_Architectures.resnet import ResNet50, ResNet101, ResNet152
from Network_Architectures.vgg import VGG19, VGG16
from Network_Architectures.efficient_net import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
import pytorch_lightning as pl

# from PythonTools.cropping_out_with_projection import cropping_out_with_projection


model_mapping = {
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'VGG19': VGG19,
    'VGG16': VGG16,
    'EfficientNetB0': EfficientNetB0,
    'EfficientNetB1': EfficientNetB1,
    'EfficientNetB2': EfficientNetB2,
    'EfficientNetB3': EfficientNetB3,
    'EfficientNetB4': EfficientNetB4,
    'EfficientNetB5': EfficientNetB5,
    'EfficientNetB6': EfficientNetB6,
    'EfficientNetB7': EfficientNetB7,
}

optimizer_mapping = {
    'adam': 'adam',
    'sgd': 'sgd',
    'rmsprop': 'rmsprop',
    'adagrad': 'adagrad',
    'adadelta': 'adadelta',
    'adamw': 'adamw'
}


class LightningResNet(pl.LightningModule):
    def __init__(
            self,
            model_type='ResNet101',
            # model_type = 'VGG19',
            # model_type = 'EfficientNetB2',
            train_ratio=0.8,
            resize_shape=50,
            batch_size=20,
            base_lr=0.01,
            num_epochs=200,
            train_path=r'F:\projections\TrajectoryPlanningIcicle\Icosaeder_',
            optimizer='adamw',
            loss_function_type="L1Loss",

            **kwargs):

        super(LightningResNet, self).__init__()

        self.save_hyperparameters()

        assert model_type in model_mapping, "Please specify a valid model architecture."
        # assert efficientnet_type in efficientnet_model_mapping, "Please specify a valid efficientnet architecture."
        assert optimizer in optimizer_mapping, "Please specify a valid optimizer."
        self.training_path = self.hparams.inputFolder
        self.validation_epoch_outputs = []



        self.dataloader_params = kwargs

        self.optimizer = self.hparams.optimizer

        self.batch_size = self.hparams.batch_size
        self.resize_shape = self.hparams.resize_shape

        model_type = self.hparams.model_type
        self.model = model_mapping[model_type](num_regression_targets=1)
        # self.model = vgg_model_mapping[vgg_type]( num_regression_targets=1)
        # self.model = efficientnet_model_mapping[efficientnet_type](num_regression_targets=1)

        self.loss_function_type = self.hparams.loss_function_type
        self.base_lr = self.hparams.base_lr
        self.num_epochs = self.hparams.num_epochs
        self.train_ratio = self.hparams.train_ratio


        if loss_function_type == 'L1Loss':
            self.criterion = {
                'train': torch.nn.L1Loss(),
                'val': torch.nn.L1Loss(),
                'test': torch.nn.L1Loss()  # Hinzugefügt
            }
        elif loss_function_type == 'MSELoss':
            self.criterion = {
                'train': torch.nn.MSELoss(),
                'val': torch.nn.MSELoss(),
                'test': torch.nn.MSELoss()  # Hinzugefügt
            }
        elif loss_function_type == 'PoissonNLLLoss':
            self.criterion = {
                'train': torch.nn.PoissonNLLLoss(),
                'val': torch.nn.PoissonNLLLoss(),
                'test': torch.nn.PoissonNLLLoss()  # Hinzugefügt
            }

        # self.dataset = Detectability_Projections_Dataset(t_voxels=self.target_voxel , calc=True, simulate=True)
        # self.datasets = self.train_test_dataset(self.dataset,val_split=1-self.train_ratio)

    def forward(self, x):
        return self.model(x)

    def compute_poisson_nll_loss(self, reco_iter, reco_target):
        return F.poisson_nll_loss(reco_iter, reco_target)

    def compute_l1(self, reco_iter, reco_target):
        return F.l1_loss(reco_iter, reco_target)

    def compute_mse(self, reco_iter, reco_target):

        return F.mse_loss(reco_iter, reco_target)

    #def train_val_dataset(self, dataset, val_split=0.027):
    #    train_size = 1 - val_split

        # Create random splits for train and validation
    #    train_idx, val_idx = train_test_split(list(range(len(dataset))), train_size=train_size, shuffle=True)

    #    datasets = {}
    #    datasets['train'] = Subset(dataset, train_idx)
    #    datasets['val'] = Subset(dataset, val_idx)
    #    datasets['test'] = Subset(dataset, val_idx)  # You can adjust this if needed
    #    return datasets

    def test_dataset(self, dataset):
        datasets = {}
        datasets['test'] = dataset
        return datasets

    def training_step(self, batch, batch_idx):
        x, y = batch
        print(batch_idx)
        # x = x.float()  # Konvertiere die Eingabe zu float
        # y = y.float()  # Konvertiere die Zielwerte zu float
        output = self.forward(x).squeeze(1)
        # output = output.float()

        loss_function_type = self.hparams.loss_function_type

        # Rufen Sie die Verlustfunktion aus der YAML-Datei ab
        loss_function = getattr(nn, loss_function_type)()
        
        loss = loss_function(output, y)
        self.log('train/loss', loss, on_epoch=True, prog_bar=True)

        if loss_function_type == 'PoissonNLLLoss':
            poisson_loss_val = self.compute_poisson_nll_loss(output, y)
            self.log('val/PoissonNLL', poisson_loss_val)
        elif loss_function_type == 'L1Loss':
            l1_loss_val = self.compute_l1(output, y)
            self.log('val/L1', l1_loss_val)
        elif loss_function_type == 'MSELoss':
            mse_loss_val = self.compute_mse(output, y)
            self.log('val/MSE', mse_loss_val)

        return loss

    def configure_optimizers(self):
        # Convert the optimizer name to the actual optimizer class
        optimizer_name = optimizer_mapping[self.optimizer]

        if optimizer_name == 'adam':
            optimizer = Adam(self.parameters(), lr=self.base_lr, betas=(0.5, 0.999))
        elif optimizer_name == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.base_lr)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(self.parameters(), lr=self.base_lr)
        elif optimizer_name == 'adagrad':
            optimizer = Adagrad(self.parameters(), lr=self.base_lr)
        elif optimizer_name == 'adadelta':
            optimizer = Adadelta(self.parameters(), lr=self.base_lr)
        elif optimizer_name == 'adamw':
            optimizer = AdamW(self.parameters(), lr=self.base_lr, betas=(0.5, 0.999), weight_decay=0.01)

        return [optimizer]

    # def on_train_epoch_end(self):
    #     print('on train epoch end wird ausklammert')

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x).squeeze(1)

        loss_function_type = self.hparams.loss_function_type
        loss_function = getattr(nn, loss_function_type)()

        val_loss = loss_function(output, y)
        self.log('val/loss', val_loss, on_epoch=True, prog_bar=True)

        if loss_function_type == 'PoissonNLLLoss':
            poisson_loss_val = self.compute_poisson_nll_loss(output, y)
            self.log('val/PoissonNLL', poisson_loss_val)
        elif loss_function_type == 'L1Loss':
            l1_loss_val = self.compute_l1(output, y)
            self.log('val/L1', l1_loss_val)
        elif loss_function_type == 'MSELoss':
            mse_loss_val = self.compute_mse(output, y)
            self.log('val/MSE', mse_loss_val)

        # Append the validation loss to the list
        self.validation_epoch_outputs.append(val_loss.item())

        return val_loss


    def test_step(self, batch, batch_idx):
        x, y = batch

        #x = x.float()  # Konvertiere die Eingabe zu float
        #y = y.float()  # Konvertiere die Zielwerte zu float
        output = self.forward(x).squeeze(1)
        #output = output.float()  # Konvertiere die Modellvorhersagen zu float

        # Rufen Sie die Verlustfunktion aus der YAML-Datei ab
        loss_function_type = self.hparams.loss_function_type
        loss_function = getattr(nn, loss_function_type)()

        loss = loss_function(output, y)
        self.log('test/loss', loss)

        if loss_function_type == 'PoissonNLLLoss':
            self.log('test/PoissonNLL',
                     self.compute_poisson_nll_loss(output, y))  # Protokollieren Sie die PoissonNLLLoss
        elif loss_function_type == 'L1Loss':
            self.log('test/L1', self.compute_l1(output, y))
        elif loss_function_type == 'MSELoss':
            self.log('test/MSE', self.compute_mse(output, y))

        #return loss

    #updated

    #def on_validation_epoch_end(self):
    #    val_loss_mean = torch.stack(self.validation_epoch_outputs).mean()
    #    self.log('MeanValidationLoss', val_loss_mean, on_epoch=True, prog_bar=True)

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #
    #     x = x.float()  # Konvertiere die Eingabe zu float
    #     y = y.float()  # Konvertiere die Zielwerte zu float
    #     output = self.forward(x)
    #     output = output.float()  # Konvertiere die Modellvorhersagen zu float
    #
    #     # Rufen Sie die Verlustfunktion aus der YAML-Datei ab
    #     loss_function_type = self.hparams.loss_function_type
    #     loss_function = getattr(nn, loss_function_type)()
    #
    #     loss = loss_function(output, y)
    #     self.log('test/loss', loss)
    #
    #     if loss_function_type == 'PoissonNLLLoss':
    #         self.log('test/PoissonNLL',
    #                  self.compute_poisson_nll_loss(output, y))  # Protokollieren Sie die PoissonNLLLoss
    #     elif loss_function_type == 'L1Loss':
    #         self.log('test/L1', self.compute_l1(output, y))
    #     elif loss_function_type == 'MSELoss':
    #         self.log('test/MSE', self.compute_mse(output, y))
    #
    #     return loss

    # Dataloader Stuff
    def setup(self, stage: str):
        self.datasets = self.test_dataset(self.dataset)
        self.test_dataset = self.datasets['test']
        print("Setup of Dataloader done")

    #def setup(self, stage: str):
    #    self.datasets = self.train_val_dataset(self.dataset, val_split=1 - self.train_ratio)
    #    self.train_dataset = self.datasets['train']
    #    self.val_dataset = self.datasets['val']
    #    self.test_dataset = self.datasets['test']
    #    print("Setup of Dataloader done")
        # self.test_dataset = self.datasets['test']

    def prepare_data(self):

        print('Prepare Dataset')
        self.dataset = Detectability_Projections_Dataset(**self.dataloader_params)

    def train_dataloader(self):
        # values here are specific to pneumonia dataset and should be changed for custom data
        # transform = transforms.Compose([
        #         transforms.Resize((self.resize_shape,self.resize_shape)),
        #         transforms.RandomHorizontalFlip(0.3),
        #         transforms.RandomVerticalFlip(0.3),
        #         transforms.RandomApply([
        #             transforms.RandomRotation(180)
        #         ]),
        #         transforms.ToTensor()])
        # self.dataset.set_transformations(transform)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)

    def val_dataloader(self):
        # values here are specific to pneumonia dataset and should be changed for custom data
        # transform = transforms.Compose([
        # transforms.Resize((self.resize_shape, self.resize_shape)),
        # transforms.ToTensor()])
        # self.dataset.set_transformations(transform)

        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=5)

    def test_dataloader(self):
         # values here are specific to pneumonia dataset and should be changed for custom data
         #transform = transforms.Compose([
         #transforms.Resize((self.resize_shape, self.resize_shape)),
         #transforms.ToTensor()])
         #self.dataset.set_transformations(transform)

        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=5)