# Human Circadian Rhythms Simulation Package in Python (hrsimpy)

Tools for simulating human circadian rhythms for a given light schedule. This package implements models for human circadian rhythms and some tools for visualizing the rhythms in python.

The background for these models can be found in our article published in the Journal of Biological Rhythms:

https://journals.sagepub.com/eprint/CZRXAPFRWA94ZMFDZWWW/full

Some things to note are the differential equations since the code relies heavily on them.

## Installing the Package

This repo is now available as a pip package. It may be installed by giving the command:

pip install hcrsimpy

The dependencies for the install are given in the file requirements.txt. You
can install the dependencies once you have downloaded the requirements.txt file by
giving:

pip install -r requirements.txt

That being said it doesn't have any exotic dependencies, most of the libraries will be installed for anyone who does scientific programming in python.

## Model interface

One of the main ideas for this package is to compile a source of human circadian models with their associated parameter values. This field has now grown to the point that many model variants exists and (I think) it is worthwhile to create a simple open-source way to simulate the circadian dynamics using a selection of models.

The available models may be imported (once the package is installed) with the command:

```{python}
from hcrsimpy.models import *
```

The list of models available can be found in the hcrsimpy/models directory in the repo above. Those files are written with the hope that they are readable by someone who has just begun learning python.

Okay great we can import some models, but the idea is to have a uniform input and output to simulate circadian dynamics using one of the available models.

All of the these models are implemented as a python class, with their parameters stored as members of that class. 

The idea to to be able to quickly change between and compare the outputs for different models. 

Here is an example run for the the single population variant of my model:

```{python}
from hcrsimpy.plots import Actogram
from hcrsimpy.models import *
from hcrsimpy.light import *

ts =  np.arange(0.0, 24*30, 0.10)
light_values = np.array([ShiftWorkerThreeTwelves(t, Intensity=150.0) for t in ts])
model = SinglePopModel()
initial_conditions = np.array([1.0,0.0,0.0])
sol = model.integrate_model(ts=ts, light_est=light_values, state=initial_conditions)
dlmo = model.integrate_observer(ts=ts, light_est=light_values, u0 = initial_conditions )
acto = Actogram(ts, light_vals=light_values, opacity=0.50)
acto.plot_phasemarker(dlmo, color='blue')
plt.title("Actogram for a Simulated Shift Worker")
plt.tight_layout()
plt.show()
```

# Command Line Tools 

The package comes with two handy command line tools for creating actograms (acto command) from 
wearable data and computing the Entrainment Signal Regularity Index (ESRI) (esri command). 

After a pip install you can use those tools straight from the command line. They each
have a whole host of command line arguments you can use and are documented if you give a

```
acto --help
```
or 
```
esri --help
```

# Model Advice and Acceleration

The Single Population model was created by me, and is the most useful one in my work. Therefore, it has 
some special features included and is the default for the acto/esri command line scripts. It also is written 
under the hood in C to enable lightning fast integration. So I would recommend using that one if you want some 
extra speed. 

# Data Formats 



# Create your own light schedules

For each of the models you can pass in and defined light schedule which gives light levels in lux. The file LightSchedules has some I have developed but you can build your own from data or otherwise.



