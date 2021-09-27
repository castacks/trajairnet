# Predicting  Like  A  Pilot:  Dataset  and  Method  to  Predict  Socially-Aware Aircraft  Trajectories  in  Non-Towered  Terminal  Airspace

This repository contains the code for the paper submitted to ICRA 2022. 

[Predicting  Like  A  Pilot:  Dataset  and  Method  to  Predict  Socially-Aware Aircraft  Trajectories  in  Non-Towered  Terminal  Airspace]() 

[Jay Patrikar](https://jaypatrikar.me/), [Brady Moon](https://bradymoon.com/), [Jean Oh](https://www.cs.cmu.edu/~./jeanoh/) and [Sebastian Scherer](https://www.ri.cmu.edu/ri-faculty/sebastian-scherer/).


## Brief Overview [[Video]]()

![ha](images/Fig1v4.pdf)

Pilots  operating  aircraft  in  un-towered  airspacerely  on  their  situational  awareness  and  prior  knowledge  topredict the future trajectories of other agents. These predictionsare conditioned on the past trajectories of other agents, agent-agent  social  interactions  and  environmental  context  such  as airport  location  and  weather.  This  paper  provides  a  dataset, TrajAir, that captures this behaviour in a non-towered terminal airspace around a regional airport. We also present a baseline socially-aware trajectory prediction algorithm, TrajAirNet, that uses  the  dataset  to  predict  the  trajectories  of  all  agents.  The dataset  is  collected  for  111  days  over  8  months  and  contains ADS-B transponder data along with the corresponding METAR weather data. The data is processed to be used as a benchmark with  other  publicly  available  social  navigation  datasets.  To the  best  of  authors’  knowledge,  this  is  the  first  3D  social aerial  navigation  dataset  thus  introducing  social  navigationfor  autonomous  aviation.TrajAirNet combines  state-of-the-artmodules  in  social  navigation  to  provide  predictions  in  a  static environment with a dynamic context. 

## Installation

### Environment Setup

First, we'll create a conda environment to hold the dependencies.

```
conda create --name trajairnet --file requirements.txt
conda activate trajairnet
```

### Data Setup

The network uses [TrajAir Dataset](https://theairlab.org/trajair/).

```
cd dataset
wget https://kilthub.cmu.edu/ndownloader/articles/14866251/versions/1
```

Unzip files as required.

##  Model Training

For training data we can choose between the 4 subsets of data labelled 7days1, 7days1, 7days1, 7days1 or the entire dataset 111_days. For example, to train with 7days1 use:  

`python train.py --dataset_name 7days1`

Training will use GPUs if available.

##  Model Testing

For training data we can choose between the 4 subsets of data labelled 7days1, 7days1, 7days1, 7days1 or the entire dataset 111_days. For example, to test with 7days1 use:  

`python test.py --dataset_name 7days1`

## TrajAir Dataset processing

This repository also contains untilities to process raw data from TrajAir dataset to produce the processed outputs. 

`python adsb_preprocess/process.py --dataset_name 7days1`





