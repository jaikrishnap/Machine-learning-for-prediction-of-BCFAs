# Optical properties of black carbon (BC) at various stages of aging: comprehensive dataset and machine learning-based prediction algorithm
TODO: write some introductory paragraph and maybe change title

## Installing required software
Running the prediction script requires a working Python interpreter with several packages installed. We recommend using [conda](https://conda.io/projects/conda/en/latest/index.html) to setup a virtual environment:
1. Follow the instructions at the [conda website](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to download and install Miniconda (or Anaconda if you prefer) for your OS.
2. Open a terminal window in the folder that contains this README and type the following to create a new virtual environment containing the required packages to run the prediction script:
   ```commandline
   conda env create -f conda_env.yml
   ```
   If you want to use your NVIDIA GPU to accelerate predictions using the Neural Network, please replace `conda_env.yml` with `conda_env_gpu.yml` in the above command.
3. To check whether the installation was successful, try running the following commands:
   ```commandline
   tobias@tobias-Laptop:~$ conda activate BCA
   (BCA) tobias@tobias-Laptop:~$ python
   Python 3.9.5 (default, Jun  4 2021, 12:28:51) 
   [GCC 7.5.0] :: Anaconda, Inc. on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import keras
   >>> quit()
   (BCA) tobias@tobias-Laptop:~$
   ```
TODO: add instructions for virtualenv + pip

## Running the script
To run the script, open a terminal window, clone this repository, and activate the virtual environment that you set up earlier:
```commandline
tobias@tobias-Laptop:~$ git clone https://github.com/jaikrishnap/Machine-learning-for-prediction-of-BCFAs.git prediction_script
Cloning into 'prediction_script'...
remote: Enumerating objects: 50, done.
remote: Counting objects: 100% (11/11), done.
remote: Compressing objects: 100% (10/10), done.
remote: Total 50 (delta 3), reused 7 (delta 1), pack-reused 39
Receiving objects: 100% (50/50), 43.13 MiB | 9.20 MiB/s, done.
Resolving deltas: 100% (11/11), done.
tobias@tobias-Laptop:~$ cd prediction_script
tobias@tobias-Laptop:~/prediction_script$ conda activate BCA
(BCA) tobias@tobias-Laptop:~/prediction_script$ 
```
If you do not have git installed on your machine, you can also download this repository as a zip file by clicking [here](https://github.com/jaikrishnap/Machine-learning-for-prediction-of-BCFAs/archive/refs/heads/main.zip).

The basic command to execute the script is
```commandline
(BCA) tobias@tobias-Laptop:~/prediction_script$ python predict.py [OPTIONS] -o out.csv in.csv
```
This would read input data from the file `in.csv` and save prediction results in the file `out.csv`. Please check the subsequent sections for more information on how to format your input files.
`[OPTIONS]` refers to the options explained in the [options section](#options). You can also run `python predict.py --help` to display a short summary.  
For example, the following command would use the neural network model instead of kernel ridge regression to predict the optical properties:
```commandline
python predict.py -m ann -o out_ann.csv in.csv
```

Note that the script supports writing output to STDOUT and reading input from STDIN by specifying `-` instead of a file name, so the following command would be equivalent to the previous one:
```commandline
python predict.py -m ann -o - - < in.csv > out.csv
```

### Input format
Input files need to be in CSV format. The following columns are required:
* `fraction_of_coating`: The fraction (in percent) of the total volume that is made up by the organic coating. Should be a real number between 0 and 99.
* `fractal_dimension`: The fractal dimension describing the morphology of the fractal aggregate.
* `wavelength`: The wavelength (in nm) for which to compute optical properties.

Additionally, at least one of the following columns is required to specify the size of the aggregate:
* `number_of_primary_particles`: The number of primary particles that make up the aggregate.
* `equi_mobility_dia`: The mobility diameter of the fractal aggregate.
* `vol_equi_radius_outer`: The radius of a sphere that has the same volume as the fractal aggregate including the organic coating.
* `vol_equi_radius_inner`: The radius of a sphere that has the same volume as the fractal aggregate excluding the organic coating.

Any order of the columns will work. At the moment, our script cannot handle missing values. Please see the `input_*.csv` files in the `samples` folder for some valid input files.

### Output format
The script will always output a CSV file that contains the unchanged columns from the input file plus the model's predictions and some derived values. Note that the columns in the output file will be the same and in the same order regardless of the input.
Please check `samples/output.csv` for the exact specification and our corresponding publication [^1] for information about the features.

### Options
* `-o`, `--out-file`: This options specifies the file into which the output is written. Specifying `-` will cause the script to write the results to STDOUT.
* `-m`, `--model`: Which model to use for predicting the optical properties. Use `ann` for the neural network and `krr` for kernel ridge regression. If you do not specify this option, the script uses kernel ridge regression.
* `--model-file`: Specifies the file to load the model from. This defaults to the model files in the `models` folder. Please see the section on [training your own models](#training-your-own-models) for information on the model files' format.
* `-r`, `--refractive-indices`: The output file contains refractive indices for black carbon and organic coating for certain wavelengths. They are read from the file you specify here or by default from `data/refractive_indices.csv`. Please check this file for the required format. If refractive indices for a certain input wavelength are not present in this file, the corrsponding columns will be empty in the output file.

[^1]: Patil et al., .... TODO

## Training your own models
TODO: section on how to train models.

## Citation
If you use this code as part of your own work, please consider citing our corresponding research paper:
```
TODO: add bibtex and/or plaintext 
```
