# CNNvsTransformerComparing CNN- and Transformer-Based Deep Learning Models for Semantic Segmentation of Remote Sensing Images
Master Thesis at the Institute for Geoinformatics, University of Münster, Germany. 

### Description
Dataset used: [ISPRS Benchmark on Semantic Labeling](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) (just used Potsdam dataset so far)

### How To

1. Prepare dataset in following folder structure:
```
|-- data
|   |-- rgb
|   |-- rgb_valid
|   |-- label
|   |-- label_valid
```
 - with ground truth images in folders `/label` having the same name as the corresponding images in folders `/rgb`.

2. Install requirements (compare `/PALMA/requirements.txt` and modules loaded in `/PALMA/train_unet.sh`; respective requirements file will be added later).
3. Train model with
```
python3 train.py --data_path /your/path/to/folder/data --output_path ./weights
```

 - By default a U-Net (alternative: `--model segformer`) is trained with 20 epochs (alternative: `--epochs integer`).

 - Furhter script arguments are listed in `/train.py` and can be shown with `python3 train.py --help`. 

4. Find your model weights in folder `./weights`.

5. Visualize and compare your models using the respective Notebooks in `/Notebooks` by changing the model paths and names and the data path.

### Acknowledgements

A lot of the code was inspired by https://github.com/suryajayaraman/Semantic-Segmentation-using-Deep-Learning.
