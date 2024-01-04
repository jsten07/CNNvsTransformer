# CNNvsTransformer: Comparing CNN- and Transformer-Based Deep Learning Models for Semantic Segmentation of Remote Sensing Images
Master Thesis at the Institute for Geoinformatics, University of Münster, Germany.

### Description

With the given code, results for the thesis mentioned above were created.
A pdf of the thesis will be added soon.

In the following some basic steps to use the code are described but might need some adaptations, based on the utilized infrastructure.

Further documentation will be added soon. 

### How To Train the Models
1. Prepare data
	1. Download data, for example one of:
		1. [ISPRS Benchmark on Semantic Labeling](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)
		2. [FloodNet](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021)
	2. Patchify data into appropriate sizes, e.g. $512\times 512$
		1. `\helper\patchify.py` might be useful
	3. Split data into train and test data in the following folder structure, with `rgb` folders containing the corresponding images to the groundtruth labels in the `label` folders:
```
|-- data
|   |-- rgb
|   |-- rgb_test
|   |-- label
|   |-- label_test
```
2. If you use the PALMA cluster, just adapt and run one of `\PALMA\train_unet.sh` or `\PALMA\train_segformer.sh` and you are done
3. Otherwise: Install requirements (see `\PALMA\requirements.txt` and modules in `\PALMA\train_unet.sh`) 
4. Run `train.py` by
```
python3 train.py --data_path /your/path/to/folder/data --name ./weights
```
5. Find your model in folder `./weights`

By default a U-Net model will be trained for 20 epochs. Further default settings can be derived from the parameters of `train.py`.

### Evaluation and Visualization

The evaluation and visualization of the models was done with help of the notebooks in the respective `./Notebooks` directory.

- `compare.ipynb`: comparison of two models by visualizing predictions of both and calculating metrics on test data
- `homogeneity.ipynb`: calculation of clustering evaluation measures
- `radarchart.ipynb`: plot radar charts and bar charts for the calculated metrics
- `count_classes.ipynb`: count pixels per class and mean and standard deviation in an image dataset
- OUTDATED: `Segformer_Run.ipynb`, `UNet_Run.ipynb`, `Segformer_visualize_attention.ipynb`


### Acknowledgements

A lot of the code was inspired by https://github.com/suryajayaraman/Semantic-Segmentation-using-Deep-Learning.
