
import numpy as np

import torchaudio
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam

from sacred import Experiment

from experiments.core import prepare_nshot_task
from experiments.core import categorical_accuracy
from models.protonet import proto_net_episode
from models.backbones.standard_backbone import get_backbone
from models.backbones.resnet_backbone import ResNet18
from dataLoader.data_iterator import SequenceDataset
from dataLoader.task_sampler import NShotTaskSampler

EXPERIMENT_NAME = 'prototypical networks experiment for voice verification'
ex = Experiment(EXPERIMENT_NAME)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print('device that will be used is', device)

class ProtoTrainer():
    def __init__(self):
        self.train_taskloader, self.test_taskloader = self.get_dataLoader()
        self.model, self.spectro = self.get_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.loss_fn = self.get_loss_fn()
    
    @ex.capture
    def get_dataLoader(self, min_seq, max_seq, downsampling, episodes_per_epoch, n_train, k_train, q_train, test_episodes_per_epoch, n_test, k_test, q_test, spectrogram):
        train_data = SequenceDataset(min_seq, max_seq, downsampling, 'train', spectrogram)
        train_taskloader = DataLoader(
            train_data,
            batch_sampler = NShotTaskSampler(train_data, episodes_per_epoch, n_train, k_train, q_train),
            num_workers = 0
        )

        test_data = SequenceDataset(min_seq, max_seq, downsampling, 'test', spectrogram)
        test_taskloader = DataLoader(
            train_data,
            batch_sampler = NShotTaskSampler(test_data, test_episodes_per_epoch, n_test, k_test, q_test),
            num_workers = 0
        )
        return(train_taskloader, test_taskloader)

    @ex.capture
    def get_model(self, spectrogram):
        if spectrogram:
            return(ResNet18().to(device, dtype=torch.float), nn.Sequential(torchaudio.transforms.Spectrogram(n_fft=255, hop_length=160)).to(device))
        else:
            return(get_backbone(input="1d", kernel=32, pad=0).to(device, dtype=torch.float)) 

    @ex.capture
    def get_optimizer(self, learning_rate):
        return(Adam(self.model.parameters(), lr=learning_rate))

    @ex.capture
    def get_scheduler(self, step_size, gamma):
        return(StepLR(self.optimizer, step_size=step_size, gamma=gamma))

    @ex.capture
    def get_loss_fn(self):
        return(torch.nn.NLLLoss())

    @ex.capture
    def train(self, epochs, n_train,k_train, q_train, episodes_per_epoch , n_test, k_test, q_test, distance, final_test_episodes, test_episodes_per_epoch, save_model, save_model_file ,_run):
        for epoch in range(1, epochs+1):
            train_accuracy = []
            test_accuracy = []
            loss_train = []

            for batch in self.train_taskloader:
                def handle_trainbatch():
                    x, y = prepare_nshot_task(batch, k_train, q_train)

                    x = x.to(device)
                    y = y.to(device)

                    loss, y_pred = proto_net_episode(self.model, self.spectro,self.optimizer,self.loss_fn,x,y,n_train,k_train,q_train,distance,True)

                    train_acc = categorical_accuracy(y, y_pred)
                    train_accuracy.append(train_acc)
                    loss_train.append(loss)

                handle_trainbatch()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            self.scheduler.step()

            for batch in self.test_taskloader:
                def handle_testbatch():
                    with torch.no_grad():
                        x, y = prepare_nshot_task(batch, k_test, q_test)

                        x = x.to(device)
                        y = y.to(device)

                        loss, y_pred = proto_net_episode(self.model, self.spectro,self.optimizer,self.loss_fn,x,y,n_test,k_test,q_test,distance,False)

                        test_acc = categorical_accuracy(y, y_pred)
                        test_accuracy.append(test_acc)

                handle_testbatch()

            acc_train = sum(train_accuracy)/episodes_per_epoch
            acc_test = sum(test_accuracy)/test_episodes_per_epoch
            metric_loss = sum(loss_train)/episodes_per_epoch
            _run.log_scalar('train accuracy', acc_train)
            _run.log_scalar('test accuracy', acc_test)
            _run.log_scalar('loss train', metric_loss)
            print('Epochs:', epoch)
            print('Train accuracy:', acc_train)
            print('Test accuracy:', acc_test)
            print('Loss train:', metric_loss)

        final_test_accuracy = []
        NUM_TEST_POINTS = int(final_test_episodes/test_episodes_per_epoch)
        for _ in range(NUM_TEST_POINTS):    
            for batch in self.test_taskloader:
                with torch.no_grad():
                    x, y = prepare_nshot_task(batch, k_test, q_test)
                
                    x = x.to(device)
                    y = y.to(device)
                
                    loss, y_pred = proto_net_episode(self.model,self.spectro,self.optimizer,self.loss_fn,x,y,n_test,k_test,q_test,distance,False)
                    test_acc = categorical_accuracy(y, y_pred)
                    _run.log_scalar('final accuracy loop', test_acc)
                    final_test_accuracy.append(test_acc)

        metaval_accuracies = np.array(final_test_accuracy)
        means = np.mean(metaval_accuracies, 0)
        stds = np.std(metaval_accuracies, 0)
        ci95 = 1.96*stds/np.sqrt(final_test_episodes)
        _run.log_scalar('final mean accuracy', means)
        _run.log_scalar('final std accuracy', stds)
        _run.log_scalar('final CI95 accuracy', ci95)
        print('final mean accuracy:', means)
        print('final std accuracy:', stds)
        print('final CI95:', ci95)

        if save_model:
            torch.save(self.model.state_dict(), save_model_file)

@ex.config
def config():
    distance = 'l2'

    n_train = 5
    k_train = 5
    q_train = 5

    n_test = 5
    k_test = 5
    q_test = 5

    episodes_per_epoch = 100
    test_episodes_per_epoch = 50
    final_test_episodes = 500

    epochs = 2
    learning_rate = 0.001
    step_size = 20
    gamma = 0.5

    min_seq = 1
    max_seq = 3
    downsampling = 4
    spectrogram = False

    save_model = False
    save_model_file = 'model.pt'

@ex.automain
def main(_run):
    """
    Sacred needs this main function, to start the experiment.
    If you want to import this experiment in another file (and use its configurations there, you can do that as follows:
    import proto_net
    ex = proto_net.ex
    Then you can use the 'ex' the same way we also do in this code.
    """
    trainer = ProtoTrainer()
    trainer.train()
