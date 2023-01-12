# Example run for forger 99 vdp model

from hcrsimpy.plots import actogram
from hcrsimpy.models import *
from hcrsimpy.light import *
import sys

sys.path.append("..")


duration = 16.0  # gets 8 hours of sleep
intensity = 150.0
wake = 6.0


def LightFunReg(t):
    return RegularLightSimple(t, intensity, wake, duration)


# Problem with vdp_model 
# maybe HCRSimPY? 
a = vdp_forger99_model(LightFunReg)
a.integrateModel(24 * 40)
tsdf = a.getTS()

plt.figure()
ax = plt.gca()
acto = actogram(ax, tsdf)  # add an actogram to those axes

plt.title('Forger 1999 VDP Entrainment under Regular Light Conditions')
plt.tight_layout()
plt.show()
