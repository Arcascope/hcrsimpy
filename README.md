# Human Circadian Rhythms Simulation Package in Python

Tools for simulating human circadian rhythms for a given light schedule. This package implements three models for human circadian rhythms and some tools for visualizing the rhythms in python.

I have (7/2019) updated this library to use Python 3. It should be compatible with python 2 although I won't be
spending much time on ensuring this in the future. 

The background for these models can be found in our article published in the Journal of Biological Rhythms:

https://journals.sagepub.com/eprint/CZRXAPFRWA94ZMFDZWWW/full

Enjoy!

## Installing the Package


Dependencies:

This package makes use of the most recent versions of the following python libraries:

1. Numpy
2. scipy
3. pylab/matplotlib
4. pandas
5. numba
6. seaborn
7. statsmodels






# Simulating Schedules:

The file compare_models_actograms contains tools for simulating the three models under basic light schedules including a Regular Light Schedule, Shift Work and Jetlag (slam shifts).

The file test_light_schedules also can be use to create some actograms for some basic schedules (regular, shift work, jetlag).

To get you up and running try the commands

python test_light_schedules.py

This will show three actograms with the CBT and DLMO times indicated. 

You can also run the program:

python compare_models_actograms.py

which will show a comparison of predictions between models. 


# Create your own light schedules

For each of the models you can pass in and defined light schedule which gives light levels in lux. The file LightSchedules has some I have developed but you can build your own from data or otherwise. 




