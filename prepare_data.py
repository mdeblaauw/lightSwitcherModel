import os
import shutil

datasets = ['train-clean-100', 'train-clean-360', 'dev-clean']
path = 'data/LibriSpeech/'
prepared_folder = 'prepared_data'
train_speakers = 0
test_speakers = 0
train_samples = 0
test_samples = 0

for dataset in datasets:
    assert os.path.isdir(os.path.join(path , dataset)), 'The {} directory does not exist or is not correctly placed'.format(dataset)

new_path = os.path.join(path, prepared_folder)

if not os.path.isdir(new_path):
    os.mkdir(new_path)

def get_info(dataset):
    dataset_path = os.path.join(path, dataset)
    
    info = []
    for root, folders, files in os.walk(dataset_path):
        if not folders:
            info.append({
                'path':root,
                'files':files
            })
    return(info)

def replace_data(dataset_path, info):
    global train_samples
    global test_samples

    for inf in info:
        id = inf['path'].split('/')[-2]

        if not os.path.isdir(os.path.join(dataset_path, id)):
            os.mkdir(os.path.join(dataset_path, id))

        path_dst = os.path.join(dataset_path,id)

        if len(inf['files']) < 11:
            continue;

        if dataset_path.split('/')[-1] == 'test':
            test_samples += len(inf['files']) - 1
        else:
            train_samples += len(inf['files']) - 1
        
        for file in inf['files']:
            if file.split('.')[-1] == 'txt':
                continue;

            path_src = inf['path']
            src = os.path.join(path_src,file)
            dst = os.path.join(path_dst,file)
            os.rename(src,dst)

def move_data(dataset, info):
    global train_speakers
    global test_speakers

    if dataset == 'dev-clean':
        test_speakers += len(info)
        dataset_path = os.path.join(new_path, 'test')
        if not os.path.isdir(dataset_path):
            os.mkdir(dataset_path)
        replace_data(dataset_path, info)
    else:
        train_speakers += len(info)
        dataset_path = os.path.join(new_path, 'train')
        if not os.path.isdir(dataset_path):
            os.mkdir(dataset_path)
        replace_data(dataset_path, info)

for dataset in datasets:
    info = get_info(dataset)
    move_data(dataset, info)

print('train_speakers is:', train_speakers)
print('train samples is:', train_samples)
print('test_speakers is:', test_speakers)
print('test_samples is :', test_samples)