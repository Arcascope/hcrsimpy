# :no_entry: [DEPRECATED] Active at https://github.com/Arcascope/circadian

## Human Circadian Rhythms Simulation Package in Python

Tools for simulating human circadian rhythms for a given light schedule. This package implements models for human circadian rhythms and some tools for visualizing the rhythms in python.

I have (7/2019) updated this library to use Python 3. It should be compatible with python 2 although I won't be
spending much time on ensuring this in the future.

The background for these models can be found in our article published in the Journal of Biological Rhythms:

https://journals.sagepub.com/eprint/CZRXAPFRWA94ZMFDZWWW/full

Some things to note are the differential equations since the code relies heavily on them. In particular, equations such as the Kronauer-Jewett Model are important and you will find this particular equation featured within the code of the twopop_model.py file. Frequently applied equations include:

* Kronauer-Jewett Light Process

* VDP Model

* Clock Neuron Model

* Cauchy (Lorentzian) Distribution

* Least-squares cost function



Enjoy!

Common dependencies/Libraries:
* numpy
* math
* matplotlib
* scipy
* pandas
* statsmodels
* pylab
* numba


Note: The code here uses the most recent and updated versions of these libraries.


## Installing the Package

This repo is now available as a pip3 package. It may be installed by giving the command:

pip3 install HCRSimPY

The dependencies for the install are given in the file requirements.txt. You
can install the dependencies once you have downloaded the requirements.txt file by
giving:

pip3 install -r requirements.txt

That being said it doesn't have any exotic dependencies, most of the libraries will be installed for anyone who does scientific programming in python.

## Model interface

One of the main ideas for this package is to compile a source of human circadian models with their associated parameter values. This field has now grown to the point that many model variants exists and (I think) it is worthwhile to create a simple open-source way to simulate the circadian dynamics using a selection of models.

The available models may be imported (once the package is installed) with the command:

```{python}
from HCRSimPY.models import *
```

The list of models available can be found in the HCRSimPY/models directory in the repo above. Those files are written with the hope that they are readable by someone who has just begun learning python.

Okay great we can import some models, but the idea is to have a uniform input and output to simulate circadian dynamics using one of the available models.

All of the these models are implemented as a python class, with their parameters stored as members of that class. They are all initialized by passing in a time dependent light function.

```{python}
from HCRSimPY.models import *
mymodel=chosenModel(LightTimeSeries)
```

All of these models have the following methods defined for them.....

* setParameters: called when the model is created, used default param values. Can be called to reset parameters to the default easily.
* updateParameters: Given a dictionary of parameters this will update the model parameter values
* getParameters: gives you a dictionary of the current parameter values.
* derv: defines the dynamical system for the ODE solvers.
* integrateModel(tend, initial): integrates the model for t=(0,tend) using the initial
values init.
* integrateModelData((tstart,tend), init, dt=0.1): This can integrate on a given time interval.
* integrateTransients(numdays=50) Integrates using the given light scehdules for a long period. Used to get rid of transients.
* getTS(): Once you have interated the model you can use this to give a pandas dataframe of the solution. This should be organized to be Time,Light_Level,Phase,R plus any other model specific columns needed. R here is the amplitude of the limit cycle oscillator.

The idea to to be able to quickly change between and compare the outputs for different models. ]

Here is an example run for the forger1999 vdp model.

```{python}
#Example run for forger 99 vdp model

import pylab as plt


from HCRSimPY.plots import *
from HCRSimPY.light_schedules import *
from HCRSimPY.models import *
from HCRSimPY.plots import actogram


duration=16.0 #gets 8 hours of sleep
intensity=150.0
wake=6.0
LightFunReg=lambda t: RegularLightSimple(t,intensity,wake,duration)

a=vdp_forger99_model(LightFunReg)
a.integrateModel(24*40)
tsdf=a.getTS()


plt.figure()
ax=plt.gca()
acto=actogram(ax, tsdf) #add an actogram to those axes

plt.title('Forger 1999 VDP Entrainment under Regular Light Conditions')
plt.tight_layout()
plt.show()

```

More examples of using the library are in the Examples directory above.

# Circadian Plots

I have implemented a actogram plotter and stoboscopic plot methods which can be plotted using the getTS method (they use that dataframe).

I would like to build some additional circadian visualizing tools in the future.

# Create your own light schedules

For each of the models you can pass in and defined light schedule which gives light levels in lux. The file LightSchedules has some I have developed but you can build your own from data or otherwise.

For measured data schedules you can pass in an interpolated light function from the data. I have implemented this for several data sets using actiwatch data. You probably will want to smooth the input before interpolating.

# Help wanted

I would welcome any help or corrections people have on creating classes for different human models. I will try to be the most commonly used ones in here first.

# Goals

* I would like to build up a nice database of models and a ensemble method for
simulating from a suite of models.
* I would also like to build a database of fit parameter values for the models


# Development Notes

Virtual environment for dev

* source hcrsim/bin/activate
* pip install -e .
* pip freeze > requirements.txt
* deactivate

# Further Documentation
Dated May/June 2022

* Updated package requirement specifications
* More comments and documentation of software available for light schedule
* Added PEP-8 compliance
* Variable renaming for clarification
* Less code ambiguity
* Restructured code in general

* Imported new packages
* Resolved issues and errors with vdp_hilaire07_model.py
* Updated vdp_hilaire07_model.py
* Resolved issues and errors with test_vdp_simple.py
* Updated test_vdp_simple.py
* Overall code readability enhanced
* Minor syntax errors resolved overall

* Various implementations of differential equations/mathematical models inspected and corrected
* Ex from prev.: Kronauer-Jewett Light Process, Clock Neuron Model, etc... 

