from circular_stats import *
import pandas as pd
import scipy as sp
import seaborn as sbn
import numpy as np
import scipy as sp
import pylab as plt
from circstats import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind
from LightSchedule import *
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.gridspec as gridspec
from latexify import *

from LightSchedule import *
from singlepop_model import *
from vdp_model import *
from twopop_model import *

convert_mil_time=lambda x: pd.DatetimeIndex(x).hour+pd.DatetimeIndex(x).minute/60.0+pd.DatetimeIndex(x).second/3600.0

def findCircularCorr(data1, criteria=None):
    """Find the Circular Correlation between the DLMO Times and the measured marker for the three models. Assume the data is given in military time"""
    if criteria is None:
        criteria=[True for i in range(0, data1.shape[0])]
    spVal=nancorrcc(hchs[criteria].SP_DLMO*sp.pi/12.0, data1*sp.pi/12.0)
    tp=nancorrcc(hchs.TP_DLMO[criteria]*sp.pi/12.0, data1*sp.pi/12.0)
    vdp=nancorrcc(hchs.VDP_DLMO[criteria]*sp.pi/12.0, data1*sp.pi/12.0)
    return((spVal, tp, vdp))



hchs2=pd.read_csv('hchs_model_diff.csv').rename(columns=lambda x: x.strip())
column_list=['PID', 'SAWA9', 'SAWA339', 'SHIFTWORKERYN', 'SAWA174', 'SAWA313', 'SAWA315', 'SAWA316', 'SAWA323', 'RMEQ', 'SAWA325', 'SAWA326', 'SAWA327', 'SAWA328', 'SAWA317']
sleep_data=pd.read_csv('../../HumanData/HCHS/datasets/hchs-sol-sueno-ancillary-dataset-0.3.0.csv', usecols=column_list)
sleep_data=sleep_data.rename(columns={'PID':'Filename','SAWA9':'Num_Good_Days', 'SAWA339':'Inter_Day_Variability', 'SAWA174':'Av_Sleep_Onset_Weekend', 'SAWA313':'Av_Bedtime', 'SAWA315':'Av_Sleep_Onset', 'SAWA316':'Sd_Sleep_Onset', 'SAWA323':'Av_MidSleep', 'SAWA325':'White_Light', 'SAWA326':'Blue_Light', 'SAWA327':'Green_Light', 'SAWA328':'Red_Light', 'SAWA317':'Av_Sleep_Offset'})


#Remove any errors from the hchs sims
hchs2=hchs2.loc[hchs2.SP_DLMO>0.0]

#Cast the times in the sleep data to the correct formats
sleep_data.Av_Sleep_Onset=pd.to_datetime(sleep_data.Av_Sleep_Onset, format='%H:%M:%S')
sleep_data.Sd_Sleep_Onset=pd.to_datetime(sleep_data.Sd_Sleep_Onset, format='%H:%M:%S')
sleep_data.Av_MidSleep=pd.to_datetime(sleep_data.Av_MidSleep, format='%H:%M:%S')
sleep_data.Av_Bedtime=pd.to_datetime(sleep_data.Av_Bedtime, format='%H:%M:%S')
sleep_data.Av_Sleep_Onset_Weekend=pd.to_datetime(sleep_data.Av_Sleep_Onset_Weekend, format='%H:%M:%S')

#Convert the times to military times
sleep_data.Av_Sleep_Onset=convert_mil_time(sleep_data.Av_Sleep_Onset)
sleep_data.Sd_Sleep_Onset=convert_mil_time(sleep_data.Sd_Sleep_Onset)
sleep_data.Av_MidSleep=convert_mil_time(sleep_data.Av_MidSleep)
sleep_data.Av_Bedtime=convert_mil_time(sleep_data.Av_Bedtime)
sleep_data.Av_Sleep_Offset=convert_mil_time(sleep_data.Av_Sleep_Offset)
sleep_data.Av_Sleep_Onset_Weekend=convert_mil_time(sleep_data.Av_Sleep_Onset_Weekend)

hchs=pd.merge(hchs2, sleep_data, on='Filename', how='left')

#Drop any rows missing simulation data for some reason
hchs=hchs[hchs['Num_Good_Days']>=5]
hchs=hchs.dropna(subset=['SP_DLMO', 'TP_DLMO', 'VDP_DLMO'])

#Add some additional columns
diff_sptp=[]
diff_spvdp=[]
diff_tpvdp=[]
diff_spSODLMO=[]
diff_vdpSODLMO=[]
diff_tpSODLMO=[]
for i in range(0, hchs.shape[0]):
    diff_sptp.append(subtract_clock_times(hchs.SP_DLMO.iloc[i], hchs.TP_DLMO.iloc[i]))
    diff_spvdp.append(subtract_clock_times(hchs.SP_DLMO.iloc[i], hchs.VDP_DLMO.iloc[i]))
    diff_tpvdp.append(subtract_clock_times(hchs.TP_DLMO.iloc[i], hchs.VDP_DLMO.iloc[i]))
    diff_spSODLMO.append(subtract_clock_times(hchs.Av_Sleep_Onset.iloc[i], hchs.SP_DLMO.iloc[i]))
    diff_tpSODLMO.append(subtract_clock_times(hchs.Av_Sleep_Onset.iloc[i], hchs.TP_DLMO.iloc[i]))
    diff_vdpSODLMO.append(subtract_clock_times(hchs.Av_Sleep_Onset.iloc[i], hchs.VDP_DLMO.iloc[i]))

hchs['diff_sptp']=pd.Series(diff_sptp, index=hchs.index)
hchs['diff_spvdp']=pd.Series(diff_spvdp, index=hchs.index)
hchs['diff_tpvdp']=pd.Series(diff_tpvdp, index=hchs.index)
hchs['diff_spSODLMO']=pd.Series(diff_spSODLMO, index=hchs.index)
hchs['diff_tpSODLMO']=pd.Series(diff_tpSODLMO, index=hchs.index)
hchs['diff_vdpSODLMO']=pd.Series(diff_vdpSODLMO, index=hchs.index)


#Find Major differences in the model predictions
criteria=(abs(hchs.diff_spvdp)>1.0) #& (abs(hchs.diff_sptp)<0.5)
criteriaInt=map(int, criteria)
hchs['Model_Diff']=pd.Series(criteriaInt, index=hchs.index)

#Save a copy of the list of model diff
list_of_discrepancies=list(hchs[criteria].Filename)
criteriaNOT=[not c for c in criteria]
list_of_agreements=list(hchs[criteriaNOT].Filename)

print "Number of Model Diff: ", sum(criteriaInt)
print "Percentage of Model Diff: ", sum(criteriaInt)/float(hchs.shape[0])*100
print hchs.shape


x=np.linspace(0.0,24.0,200)

data_list=[]
for f in list_of_discrepancies:
    try:
        f=str(f)
        if len(f)!=8:
            while (len(f)<8):
                f='0'+f
    
        filename='../../HumanData/HCHS/hchs-sol-sueno-'+f+'.csv'
        ls=hchs_light(filename)
        av_data=ls.data.groupby(by=['TimeCount']).mean()
        lf=InterpolatedUnivariateSpline(av_data.index, av_data.Lux, k=1)
        y_vals=map(lf,x)
        for i in range(len(x)):
            data_list.append((str(f), 'Disagree', x[i], LightLog(y_vals[i])))
    except:
        print "Error with file: ", f

for f in list_of_agreements:
    try:
        f=str(f)
        
        if len(f)!=8:
            while (len(f)<8):
                f='0'+f
    
        filename='../../HumanData/HCHS/hchs-sol-sueno-'+f+'.csv'
        ls=hchs_light(filename)
        av_data=ls.data.groupby(by=['TimeCount']).mean()
        lf=InterpolatedUnivariateSpline(av_data.index, av_data.Lux, k=1)
        y_vals=map(lf,x)
        for i in range(len(x)):
            data_list.append((str(f), 'Agree', x[i], LightLog(y_vals[i])))
    except:
        print "Error with file (agreement): ", f


lightSchedulesD=pd.DataFrame(data_list, columns=['PID', 'Agreement', 'Time', 'Log_Lux'])        
        
latexify(columns=1)        
plt.figure()
G = gridspec.GridSpec(1, 2)
ax2= plt.subplot(G[0, 0])
ax1=plt.subplot(G[0, 1])


sbn.lineplot(x="Time", y="Log_Lux", data=lightSchedulesD, hue='Agreement', style='Agreement', ci=None, lw=2.0, ax=ax1, legend=False);
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles=handles[1:], labels=labels[1:])
#ax1.set_yscale('log', basex=10)
ax1.set_xlabel('Time of Day');
ax1.set_ylabel(r'$\log_{10}(Lux)$');
#ax1.set_title('Light Schedules and Model Discrepancies');
ax1.set_xlim(0,24.5)
ax1.text(1.5, 3.5, '(b)')
ax1.set_xticks([0,6,12,18,24])


#Add a Regression plot to show differences in the predictions
sbn.regplot('SP_DLMO', 'VDP_DLMO', data=hchs[criteriaNOT], color='green', fit_reg=False, marker='o', scatter_kws={"s": 7, "facecolor":'none', "alpha":0.1}, ax=ax2);
sbn.regplot('SP_DLMO', 'VDP_DLMO', data=hchs[criteria], color='blue', fit_reg=False, marker='x', scatter_kws={"s": 7,  "alpha":0.1}, ax=ax2);
ax2.set_xlabel('SP DLMO Time')
ax2.set_ylabel('VDP DLMO Time')
ax2.set_xlim(15,24)
ax2.set_ylim(15,24)
ax2.plot(np.linspace(15,24,100), np.linspace(15,24,100), lw=2.0, color='red')
ax2.set_xticks([15,18,21,24])
#ax2.set_title('SP DLMO versus VDP DLMO');
ax2.text(15.5,23.0,'(a)')

plt.tight_layout()
plt.savefig('../../Figures/model_diff.eps', dpi=1200)
plt.show()


def findKeyDLMOTimes(tsdf):
    """Find the DLMO and CBT times for a given time series prediction"""

    wrapped_time=np.round(map(lambda x: fmod(x, 24.0), list(tsdf.Time)),2)
    df=pd.DataFrame({'Time': wrapped_time, 'Phase': tsdf.Phase})
    df2=df.groupby('Time')['Phase'].agg({'Circular_Mean':circular_mean, 'Phase_Coherence': phase_coherence, 'Samples':np.size})
    mean_func=sp.interpolate.interp1d(np.array(df2['Circular_Mean']), np.array(df2.index))
    return(mean_func(1.309))



def record_diff(tsdfS, tsdfV, tsdfT):
     """Find the differences in the DLMO timing of the three models for that given light schedule"""

     d1=findKeyDLMOTimes(tsdfS)
     d2=findKeyDLMOTimes(tsdfV)
     d3=findKeyDLMOTimes(tsdfT)

     return((d1,d2,d3))
    

#Find the diff for the average agree light schedule
agreeOnlyMeans=lightSchedulesD.loc[lightSchedulesD['Agreement']=='Agree'].groupby(by=['Time'])['Log_Lux'].mean()
lfa1=InterpolatedUnivariateSpline(agreeOnlyMeans.index, np.power(10, agreeOnlyMeans.values), k=1)
lfa=lambda t: lfa1(fmod(t,24.0)) #wrap the time

trans_days=50
init=guessICData(lfa, 0.0, length=trans_days)
initVDP=guessICDataVDP(lfa, 0.0, length=trans_days)
initTwo=guessICDataTwoPop(lfa, 0.0, length=trans_days)
     
a=SinglePopModel(lfa)
b=vdp_model(lfa)
c=TwoPopModel(lfa)
ent_angle=a.integrateModelData((0.0, 40.0*24.0), initial=init);
ent_angle_vdp=b.integrateModelData((0.0, 40.0*24.0), initial=initVDP);
ent_angle_two=c.integrateModelData((0.0, 40.0*24.0), initial=initTwo);
tsdf=a.getTS()
tsdf_vdp=b.getTS()
tsdf_two=c.getTS()

print "Average DLMO diff for Agree: ", np.array(record_diff(tsdf, tsdf_vdp, tsdf_two))

#Add the diff for average disagree schedule

disagreeOnlyMeans=lightSchedulesD.loc[lightSchedulesD['Agreement']=='Disagree'].groupby(by=['Time'])['Log_Lux'].mean()
lfd1=InterpolatedUnivariateSpline(disagreeOnlyMeans.index, np.power(10,disagreeOnlyMeans.values), k=1)
lfd=lambda t: lfd1(fmod(t,24.0)) #wrap the time

trans_days=50
init=guessICData(lfd, 0.0, length=trans_days)
initVDP=guessICDataVDP(lfd, 0.0, length=trans_days)
initTwo=guessICDataTwoPop(lfd, 0.0, length=trans_days)
     
a=SinglePopModel(lfd)
b=vdp_model(lfd)
c=TwoPopModel(lfd)
ent_angle=a.integrateModelData((0.0, 40.0*24.0), initial=init);
ent_angle_vdp=b.integrateModelData((0.0, 40.0*24.0), initial=initVDP);
ent_angle_two=c.integrateModelData((0.0, 40.0*24.0), initial=initTwo);
tsdf=a.getTS()
tsdf_vdp=b.getTS()
tsdf_two=c.getTS()

print "Average DLMO diff for Disagree: ", np.array(record_diff(tsdf, tsdf_vdp, tsdf_two))
    
