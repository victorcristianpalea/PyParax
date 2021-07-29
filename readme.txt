# PyParax

Numerical solver for wave propagation in the paraxial approximation

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. The software is supported on Windows 10 and Ubuntu v19.10 or greater operating systems.

### Prerequisites (Windows 10)

Use [Anaconda installer](https://www.anaconda.com/products/individual) for Windows based on the scenario that suits your institutional case. The 64-bit version is recommended.

From the Anaconda Navigator check if the following Python 3.X packages are installed (and install them if necessary):
```
 - Numpy
 - Scipy
 - Matplotlib
 - Mayavi (might require Python 3.7 to work)
```
The installation procedure should be carried using Anaconda Navigator.

The Mayavi package might need Python 3.7 and in this scenario a specific environment must be created in Anaconda Navigator with this specific version. Follow [this](https://docs.anaconda.com/anaconda/navigator/tutorials/use-multiple-python-versions/) tutorial to create the required environment.

### Prerequisites (Ubuntu)

Make sure Python 3.6 or newer is installed by typing in terminal
```
$ python3
```
The version should appear above the prompt inside the Python console as follows:
```
COMPLETEAZA DIN UBUNTU
```
Install the required packages using:
```
$ pip3 install numpy scipy matplotlib mayavi
```
### Installing
Go to the folder that contains the package files and run the command:
```
$ python3 setup.py install
```
At the end there should appear the following message:
```
Installed c:\users\victor\anaconda3\lib\site-packages\pyparax-0.0.1-py3.7.egg
Processing dependencies for pyparax==0.0.1
Finished processing dependencies for pyparax==0.0.1 
```
The installation should be carried out using Python 3.X so based on the labeling used to call the python3 command the installation procedure might require using
```
$ python setup.py install
```
Once installed, the package can be tested by importing it into a Python 3 console as follows:
```
$ python3
> import pyparax
```
## Running the tests

Inside the package folder there is a folder labeled *examples*, inside of which there is a collection of files each of which containing functions that can be used to either test the functionality, and showcase some basic usage of the package.
For basic propagation and functionality tests, the files *examples_basic_1d.py* and *examples_basic_2d.py* can be used. Inside an IDE such as Spyder open the *examples_basic_1d.py* file and run it by pressing F5, which should return
```
runfile('/location/of/the/file.py', wdir='/location/of/the/folder/in/which/the/file/is')
```
Next, call the first function *propagate_forward_freespace* in the console as 
```
> propagate_forward_freespace()
```
which should return
```
[['normal', 'normal']]
Propagate for 300 units [mm]
```
and a plot of the intensity profile for the calculated beam.
Repeat this procedure for the rest of the functions in order to test all the features of the package. 
Further explanations of what each example function does is given inside the functions.

## Authors

* **Victor-Cristian Palea**
* **Liliana Preda**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This package has been created by employees at POLITEHNICA University of Bucharest

