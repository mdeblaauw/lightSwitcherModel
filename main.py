import os
import argparse

from experiments.proto_net import ex
from sacred.observers import FileStorageObserver

added_source_files = ['models/', 'models/backbones/', 'dataLoader/']

for folder in added_source_files:
    for file in os.listdir(folder):
        if file.split('.')[-1] == 'py':
            ex.add_source_file(filename=os.path.join(folder, file))

parser = argparse.ArgumentParser()
parser.add_argument("--distance", default="l2", help="sets distance")
parser.add_argument("--n_train", default=5, type=int)
parser.add_argument("--k_train", default=5, type=int)
parser.add_argument("--q_train", default=5, type=int)
parser.add_argument("--n_test", default=5, type=int)
parser.add_argument("--k_test", default=5, type=int)
parser.add_argument("--q_test", default=5, type=int)
parser.add_argument("--train_episodes", default=100, type=int)
parser.add_argument("--test_episodes", default=25, type=int)
parser.add_argument("--final_episodes", default=500, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--step_size", default=20, type=int)
parser.add_argument("--gamma", default=0.5, type=float)
parser.add_argument("--min_seq", default=1, type=int)
parser.add_argument("--max_seq", default=3, type=int)
parser.add_argument("--downsampling", default=4, type=int)
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--save_model", help="saves the model", action="store_true")
parser.add_argument("--file_name", default="model.pt", help="name of the saved model")
parser.add_argument("--exp_name", default="my_run", help="name of experiment")
parser.add_argument("--spectrogram", help="decides which model to run", action="store_true")

args = parser.parse_args()

ex.observers.append(FileStorageObserver('experiment_results/' + args.exp_name))

print(args.spectrogram)

r = ex.run(config_updates={
    'distance':args.distance, 
    'n_train':args.n_train,
    'k_train':args.k_train,
    'q_train':args.q_train,
    'n_test':args.n_test,
    'k_test':args.k_test,
    'q_test':args.q_test,
    'episodes_per_epoch':args.train_episodes,
    'test_episodes_per_epoch':args.test_episodes,
    'final_test_episodes':args.final_episodes,
    'epochs':args.num_epochs,
    'learning_rate':args.lr,
    'step_size':args.step_size,
    'gamma':args.gamma,
    'min_seq':args.min_seq,
    'max_seq':args.max_seq,
    'downsampling':args.downsampling,
    'save_model':args.save_model,
    'save_model_file':args.file_name,
    'spectrogram':args.spectrogram})