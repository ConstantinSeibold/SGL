# Experiments on MNIST-Bags
## Experimental Setup
We follow Ilse et al. to evaluate our method for a MIL-setting. A bag is created grayscale MNIST-images of size 28 × 28, which are resized to 32 × 32. A bag is considered positive if it contains the label “9”. The number of images in a bag is Gaussian-distributed based on a fixed bag size. We investigate different average bag sizes and amounts of training bags. During evaluation 1000 bags created from the MNIST test set of the same bag size as used in training. We average the results of ten training procedures.

## Requirements
Following packages are used:
```
numpy
csv
torch
sklearn
torchvision
tqdm
time
```

Create the exact setup using:
```
conda env create -f requirements.txt
```

## Running Experiments
To run an experiment for a certain bag-size over a set of [50,100,150,20,300,500] training bags you can either run:
```
source sample_run
```
or
```
python3 mnist_exp.py --method [max,mean,mmm,bil,sgl] --instances [n_instances] 
```
