# Flora_and_Fauna_CNN_Classificatrion
A DNN Approach to Flora and Fauna Image Classification


## Enviroment

Create the vituel enviroment for this repository: 
```bash
conda create env python=3.11
```

Activate the virtual enviroment:
```bash
conda activate env
```

Install the needed packages in your enviroment:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```


## Training

For regular training with just center crop:

```bash
python train.py
```

For training with augmentation (random crop, resize, color jitter):

```bash
python train_with_aug.py
```

## Testing

For testing exchange the first argument for the path to your models weights, second argument is the name the results will be saved to:
```bash
python test.py .\training_with_augmentation\best_model.pth test_test.csv
```

## Validation

Validation is found in validation.ipynb.