# Example run for forger 99 vdp model

from hcrsimpy.plots import Actogram
from hcrsimpy.models import *
from hcrsimpy.light import *

ts =  np.arange(0.0, 24*30, 0.10)
light_values = np.array([RegularLight(t, Intensity=1500.0) for t in ts])
model = Forger99Model()
model2 = SinglePopModel()
initial_conditions = np.array([1.0,0.0,0.0])
sol = model.integrate_model(ts=ts, light_est=light_values, state=initial_conditions)
dlmo = model.integrate_observer(ts=ts, light_est=light_values, u0 = initial_conditions)
dlmo2 = model2.integrate_observer(ts=ts, light_est=light_values, u0 = initial_conditions)
acto = Actogram(ts, light_vals=light_values, opacity=0.50)
acto.plot_phasemarker(dlmo, color='blue')
acto.plot_phasemarker(dlmo2, color='darkgreen')
plt.title("Actogram for a Simulated Shift Worker")
plt.tight_layout()
plt.show()


