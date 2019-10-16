# lightSwitcherModel

## Instructions

### Requirements
Make a new venv and install requirements from `requirements.txt`

### Data

Get training data from here: <http://www.openslr.org/12>

* train-clean-100.tar.gz
* train-clean-360.tar.gz
* dev-clean.tar.gz

The train zips function as training data and the dev zip as test data.

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
