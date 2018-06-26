## Unet based model training and inference
This folder contains code for sustainbale farming practices detection using Unet.

## Prerequisites
- Python 3.5+
- [Imgaug](https://github.com/aleju/imgaug)
- Tensorflow 1.6+
- Keras 2.0.8+

## Code structure

- main.py

This is the entry point, supports _--runmode_ 'train' and 'test':
### Training
`python main.py --runmode=train --lossmode=bce --cudadev=0 --runprefix=tt3`
### Testing
`python main.py --runmode=test --lossmode=bce --cudadev=0 --runprefix=tt3`

- unet_config.py

Confiuration for training and testing the model. Here is the place to provide file path to the data, images dimentions, number of epocs and etc.

- trainer.py

Instantiates the model and trains for the specified number of steps\epochs.

- model.py

Here is definition of the Unet model.

- model_metrics.py

This file contains helper functions for model evaluation.

- losses.py

The place were custom loss functions (Dice coefficient based) reside.

- dataset_helper.py and dataset_reader.py

Here all data input related routines go.

- scorer.py

This file has heper functions to get predictions from the trained model.