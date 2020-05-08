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
