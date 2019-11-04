# lightSwitcherModel

This repository is a setup to do experiments into voice verification using deep meta-learning and, especially deep metric-learning models. you can do experiments that are automatically organised with [Sacred](https://github.com/IDSIA/sacred) and visualised with [Omniboard](https://github.com/vivekratnavel/omniboard). The goal of these experiments is to create a voice verification model that can be used to verify users of a voice assistant application. At this moment, a voice assistant PoC is created and visible in this [repo](https://github.com/mdeblaauw/lightSwitcher).

## Instructions

### Requirements

#### MongoDB Atlas
You can also use MongoDB Atlas to store experimental results. A requirement is then to set this up at <https://cloud.mongodb.com>:

1. Make a cluster and a database in the cluster.
2. Get a connection string with 'Connect to your application'.
3. Replace the connection string url and database name in `main.py`.
4. Place the login credentials in a self-created `mongodb_setup.txt` file. The structure of this file is presented below. Finally, place the file in the root folder.

`Mongodb_setup.txt` file structure:
```
username=<username>
password=<password>
```

The `main_mongo.py` script runs experiments using MongoDB Atlas. So, you should run this when you use MongoDB Atlas.

Whenever you do not use MongoDB, the experimental results are stored locally in the `experiment_results`. You do need to run the `main.py` to do this.

#### Install Python requirements
Make a new venv and install requirements from `requirements.txt`. Make sure that the Pytorch version in `requirements.txt` is compatible with your CUDA version if you want to use GPU(s).

### Data
There are two data sources from which you can do experiments: VoxCeleb or LibriSpeech.

#### VoxCeleb
To use the VoxCeleb data, we use Voxceleb2 as train set and Voxceleb1 as test set.

Get the training data from here: [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/). Example, use `curl -O -u <username> <link file>'` for mac or use wget for Linux. The username and password can be requisted from the authors of VoxCeleb.

Place the unzipped data into a `data\VoxCeleb` folder such that the file structure is as follows:

```
data/
    VoxCeleb/
        prepared_data/
            train/
                aac/
            test/
                wav/
```

The VoxCeleb2 data is formatted in m4a which can, at the moment, not be handled by torchaudio. So you can use the `m4atowav.sh` script to convert the files to WAV format. Note: this can take a while and the data will increase from approximately 80GB to 270GB. The `m4atowav.sh` script first convert all m4a files to WAV and then removes the m4a audio files.

Finally, to use the VoxCeleb data in experiments, change the `DATA_PATH` variable in `config.py` to `'data/VoxCeleb/prepared_data'`.

#### LibriSpeech
TODO: change prepare_data.py such that data_iterator can handle both VoxCeleb and LibriSpeech.

Get training data from here: <http://www.openslr.org/12>.

* train-clean-100.tar.gz
* train-clean-360.tar.gz
* dev-clean.tar.gz

The train-zips function as training data and the dev zip as test data.

Place the unzipped training data into the `data\` folder such that the file structure is as follows:

```
data/
    LibriSpeech/
        dev-clean/
        train-clean-100/
	train-clean-360/
```

Then run the `prepare_data.py` file. This creates a train and test set in `data/LibriSpeech/prepared_data/`. The class and sample distribution is as follows:

train_speakers is: 2682                                                                                                  │
train samples is: 132501                                                                                                 │
test_speakers is: 97                                                                                                     │
test_samples is : 2682  

### Omniboard

To visualise the experiments stored in MongoDB you can use Omniboard. A quickstart installation can be found at <https://github.com/vivekratnavel/omniboard/blob/master/docs/quick-start.md>.

### Run experiments
TODO
