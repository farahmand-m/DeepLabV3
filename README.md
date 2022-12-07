# Transfer Learning for Semantic Segmentation using PyTorch DeepLab v3

This repository is a modified version of the tutorial found [here](https://towardsdatascience.com/transfer-learning-for-segmentation-using-deeplabv3-in-pytorch-f770863d6a42?sk=b403331a7b30c02bff165a93823a5524) for fientuning [DeepLabV3 ResNet101](https://arxiv.org/abs/1706.05587). I needed to add a better evaluation metric to the code.

### Installing dependencies

#### Using pip

```
pip install -r requirements.txt
```

#### Using conda
```
conda env create -f environment.yml
```

### Usage of the module

```
Usage: main.py [OPTIONS]

Options:
  --data-directory TEXT  Specify the data directory.  [required]
  --exp_directory TEXT   Specify the experiment directory.  [required]
  --epochs INTEGER       Specify the number of epochs you want to run the
                         experiment for. Default is 25.

  --batch-size INTEGER   Specify the batch size for the dataloader. Default is 4.
  --help                 Show this message and exit.
```

To run the code with the `CrackForest` dataset and store the results in folder called `CFExp` use the following command.

```
python main.py --data-directory CrackForest --exp_directory CFExp
```

The datahandler module has two functions for creating datasets fron single and different folders.

By default, it uses the following:

```
def get_dataloader_single_folder(data_dir: str,
                             image_folder: str = 'Images',
                             mask_folder: str = 'Masks',
                             fraction: float = 0.2,
                             batch_size: int = 4)
```

which expects the following directory structure:

```
--data_dir
------Images
---------Image1
---------ImageN
------Masks
---------Mask1
---------MaskN
```

The repository also contains a JupyterLab file with the loss and metric plots as well as the sample prediction code.
