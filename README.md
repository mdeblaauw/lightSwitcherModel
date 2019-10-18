# lightSwitcherModel

This repository is a setup to do experiments into voice verification using deep meta-learning and, especially deep metric-learning models. The experiments are automatically organised with [Sacred](https://github.com/IDSIA/sacred) and visualised with [Omniboard](https://github.com/vivekratnavel/omniboard).

## Instructions

### Requirements
MongoDB Atlas is used to store experimental results. Hence, a requirement is to set this up at <https://cloud.mongodb.com>:

1. Make a cluster and a database in the cluster.
2. Get a connection string with 'Connect to your application'.
3. Replace the connection string url and database name in `main.py`.
4. Place the login credentials in a self-created `mongodb_setup.txt` file. The structure of this file is presented below. Finally, place the file in the root folder.

`Mongodb_setup.txt` file structure:
```
username=<username>
password=<password>
```

Make a new venv and install requirements from `requirements.txt`. Make sure that the Pytorch version in `requirements.txt` is compatible with your CUDA version if you want to use GPU(s).

### Data

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
