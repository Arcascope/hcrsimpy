"""
This class can be used to make a stroboscopic plot of the entrainment of an oscillator to a sudden shift in schedule
"""

from builtins import object
from builtins import range

import numpy as np
import scipy as sp


class stroboscopic(object):
    
    def __init__(self, ax, tsdf):
        """
        Pass in axes and a time series data frame for the model. Make sure to pass the pandas data frame starting with 
        the row you want to begin the stroboscopic plot
        """
        self.tsdf = tsdf
        self.ax = ax
        
        self.makeStroboPlot()
    
    def makeStroboPlot(self):
        """Add the paths to a quiver plot"""
        start_amp = self.tsdf.R.iloc[0]
        
        Xvals = np.array(self.tsdf.R / start_amp * sp.cos(self.tsdf.Phase))
        Yvals = np.array(self.tsdf.R / start_amp * sp.sin(self.tsdf.Phase))
        
        circle_angles = np.linspace(0, 2 * sp.pi, 1000)
        circle_x = list([sp.cos(x) for x in circle_angles])
        circle_y = list([sp.sin(x) for x in circle_angles])
        
        self.ax.plot(circle_x, circle_y, lw=2.0, color='k')
        # Sample down to every 24 hours
        Xvals = Xvals[::240]
        Yvals = Yvals[::240]
        upper_bound = min(10, len(Xvals))
        for i in range(1, upper_bound + 10):
            self.ax.quiver(Xvals[i - 1], Yvals[i - 1], Xvals[i] - Xvals[i - 1], Yvals[i] - Yvals[i - 1],
                           scale_units='xy', angles='xy', scale=1, color='blue')
        self.ax.set_xlim([-1.1, 1.1])
        self.ax.set_ylim([-1.1, 1.1])
        self.ax.scatter([0.0], [0.0], color='k')
        self.ax.set_axis_off()
    
    def addStroboPlot(self, tsdf2, col='darkgreen'):
        """Add a strobo plot to the axes for comparison"""
        start_amp = tsdf2.R.iloc[0]
        
        Xvals = np.array(tsdf2.R / start_amp * sp.cos(tsdf2.Phase))
        Yvals = np.array(tsdf2.R / start_amp * sp.sin(tsdf2.Phase))
        
        # Sample down to every 24 hours
        Xvals = Xvals[::240]
        Yvals = Yvals[::240]
        
        upper_bound = min(10, len(Xvals))
        for i in range(1, upper_bound + 10):
            self.ax.quiver(Xvals[i - 1], Yvals[i - 1], Xvals[i] - Xvals[i - 1], Yvals[i] - Yvals[i - 1],
                           scale_units='xy', angles='xy', scale=1, color=col)
