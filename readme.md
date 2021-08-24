# PyParax

Numerical solver for wave propagation in the paraxial approximation

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. The software is supported on Windows 10 and Ubuntu v19.10.

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
### Installing (Windows 10)
If using Anaconda, open Anaconda Prompt and go to the folder that contains the package files. Check the command for calling a python interpreter e.g. 
```
c:\users\your_name> python
```
and check the version that is returned after the interpreter is loaded.
```
Python 3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```
If the version is the one that you have in the Anaconda environment, then run the command:
```
$ python setup.py install
```
After installing the last lines of output text should be:
```
Installed c:\users\your_name\anaconda3\lib\site-packages\pyparax-0.0.1-py3.8.egg
Processing dependencies for pyparax==5.0
Finished processing dependencies for pyparax==5.0 
```
Once installed, the package can be tested by starting a Python interpreter and typing
```
>>> import pyparax
```
If no error is returned, PyParax should be up and running.
### Installing (Ubuntu)
Same steps as for Windows 10 but the installation is done using the Terminal.

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

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This package has been created by employees at POLITEHNICA University of Bucharest

