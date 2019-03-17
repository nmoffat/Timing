
'''
Created on 15 Sep 2016
@author: Dima Maneuski

'''

import matplotlib

matplotlib.use('TkAgg')
#from LGAD_Analysis2.WaveformAnalysisV8_4N import WaveformAnalysisV8_4N

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
import glob
import tifffile
from tifffile import TiffWriter

import thread
from time import sleep
import WaveformAnalysis_Timing_V1
#from .file import WaveformAnalysis_Timing_V1

#myDir = '/Users/neilmoffat/Documents/17060100_3249_7_7_Alpha1748/'
#myDir = 'Y:\\data\\detdev\\hvcmos01\\data\\LFoundryA\\16051900XRaysNb\\'
#myDir = '/Users/neilmoffat/Documents/Micron_diodes/run4/CNM/X-Ray/53F/'
#myDir = '/Users/neilmoffat/Documents/Micron_diodes/run4/CNM/Alpha/7859-W1_LGAD2-1/'
#myDir = '/Users/neilmoffat/Documents/Micron_diodes/run4/X-ray_variable/18290100_Variable_x_ray_40D/'
#myDir = '/Users/neilmoffat/Documents/Micron_diodes/run4/CNM/Beta/47F/'
myDir = '/Users/neilmoffat/Documents/Timing_Measurements/CERN/LGAD/CERN_Setup_190918/-10C/180V/'
obj = os.listdir(myDir)
obj.sort()

for _file in obj:
    #if _file.endswith("-5V.txt"):
    
    if _file.startswith("All_C3"): # mtrx_129_out_19_Nb
        if _file.endswith("all_waveforms_ordered.txt"):
    #if "[20_50]Sn_Bias-20V" in _file:
           
            #WaveformAnalysis_Timing_V1.WriteTiff(myDir, _file)
            WaveformAnalysis_Timing_V1.Time_resolution(myDir, _file)
            #WaveformAnalysis_Timing_V1.PlotWaveforms(myDir, _file)
            ### these are params for TJ
            #WaveformAnalysisV8_2N.WaveformAnalysisV8_2N(myDir, _file,    xLimitMinNoise = 0.0, xLimitMaxNoise = 0.005, xLimitMinSignal = 0.0, xLimitMaxSignal = 0.90, xLimitMinRiseTime = 0, xLimitMaxRiseTime = 500)


            ### these are params for LFA
            #WaveformAnalysis_Timing_V1.LoadFromFile(myDir, _file)
            #WaveformAnalysis_Timing_V1.WaveformAnalysis_Timing_V1(myDir, _file, xLimitMinNoise = -0.01, xLimitMaxNoise = 0.01, xLimitMinSignal = -0.05, xLimitMaxSignal =0.02, xLimitMinRiseTime = 0, xLimitMaxRiseTime = 9e-9)
            #WaveformAnalysisV8_4N.PlotWaveforms(myDir, _file)
            
            
#WaveformAnalysisV2('D:\\data\\Chess1\\16030400XraysFluorescence\\', 'APA1_OUT5_Cu0_0.txt')
#WaveformAnalysisV2('D:\\data\\Chess1\\16030400XraysFluorescence\\', 'APA1_OUT5_Fe0_0.txt')      
    
#WaveformAnalysisV2('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA4_OUT5_Nb0_0.txt')
#WaveformAnalysisV2('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA4_OUT5_Sn0_0.txt')
#WaveformAnalysisV2('D:\\data\\Chess1\\16030400XraysFluorescence\\', 'APA4_OUT5_Cu0_0.txt')
#WaveformAnalysisV2('D:\\data\\Chess1\\16030400XraysFluorescence\\', 'APA4_OUT5_Fe0_0.txt')

#WaveformAnalysisV2('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA7_OUT5_Cu0_0.txt')
#WaveformAnalysisV2('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA7_OUT5_Fe0_0.txt')
#WaveformAnalysisV2('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA7_OUT5_Nb0_0.txt')
#WaveformAnalysisV2('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA7_OUT5_Sn0_0.txt')

#WaveformAnalysisV2('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA8_OUT5_Cu0_0.txt')
#WaveformAnalysisV2('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA8_OUT5_Fe0_0.txt')
#WaveformAnalysisV2('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA8_OUT5_Nb0_0.txt')

#WaveformAnalysisV3('D:\\data\\Chess1\\16030400XraysFluorescence\\', 'APA5_OUT5_Fe0_0.txt')
#WaveformAnalysisV3('D:\\data\\Chess1\\16030400XraysFluorescence\\', 'APA5_OUT5_Cu0_0.txt')
#WaveformAnalysisV3('D:\\data\\Chess1\\16030400XraysFluorescence\\', 'APA5_OUT5_Nb0_0.txt')
#WaveformAnalysisV3('D:\\data\\Chess1\\16030400XraysFluorescence\\', 'APA5_OUT5_Sn0_0.txt')

#WaveformAnalysisV2('D:\\data\\Chess1\\16030400XraysFluorescence\\', 'APA6_OUT5_Fe0_0.txt')
#WaveformAnalysisV2('D:\\data\\Chess1\\16030400XraysFluorescence\\', 'APA6_OUT5_Cu0_0.txt')
#WaveformAnalysisV2('D:\\data\\Chess1\\16030400XraysFluorescence\\', 'APA6_OUT5_Nb0_0.txt')
# WaveformAnalysisV2('D:\\data\\Chess1\\16030400XraysFluorescence\\', 'APA6_OUT5_Sn0_0.txt')

#WaveformAnalysisNoise('D:\\data\\Chess1\\16030400XraysFluorescence\\', 'APA4_OUT5_Fe0_0.txt')
#WaveformAnalysisNoise('D:\\data\\Chess1\\16030400XraysFluorescence\\', 'APA4_OUT5_Cu0_0.txt')

#WaveformAnalysisNoise('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA7_OUT5_Fe0_0.txt')
#WaveformAnalysisNoise('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA7_OUT5_Cu0_0.txt')
#WaveformAnalysisNoise('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA7_OUT5_Nb0_0.txt')
#WaveformAnalysisNoise('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA7_OUT5_Sn0_0.txt')

#WaveformAnalysisNoise('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA8_OUT5_Fe0_0.txt')
#WaveformAnalysisNoise('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA8_OUT5_Cu0_0.txt')
#WaveformAnalysisNoise('D:\\data\\Chess1\\16030700XraysFluorescence\\', 'APA8_OUT5_Nb0_0.txt')

#WaveformAnalysisNoise('D:\\data\\Chess1\\16031100XraysFluorescence\\', 'APA5_OUT5_Nb0_0.txt')
#WaveformAnalysisNoise('D:\\data\\Chess1\\16031100XraysFluorescence\\', 'APA1_OUT5_Sn0_0.txt')

#WaveformAnalysisV2('D:\\data\\Chess1\\16031100XraysFluorescence\\', 'APA1_OUT5_Nb0_0.txt')
#WaveformAnalysisV2('D:\\data\\Chess1\\16031100XraysFluorescence\\', 'APA1_OUT5_Sn0_0.txt')

#WaveformAnalysisV3('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_00_0.txt')
#WaveformAnalysisV3('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_100_0.txt')
#WaveformAnalysisV3('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_200_0.txt')
#WaveformAnalysisV3('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_300_0.txt')
#WaveformAnalysisV3('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_400_0.txt')
#WaveformAnalysisV3('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_500_0.txt')
#WaveformAnalysisV3('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_600_0.txt')
#WaveformAnalysisV3('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_700_0.txt')

#WaveformAnalysisNoise('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_00_0.txt')
#WaveformAnalysisNoise('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_100_0.txt')
#WaveformAnalysisNoise('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_200_0.txt')
#WaveformAnalysisNoise('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_300_0.txt')
#WaveformAnalysisNoise('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_400_0.txt')
#WaveformAnalysisNoise('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_500_0.txt')
#WaveformAnalysisNoise('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_600_0.txt')
#WaveformAnalysisNoise('D:\\data\\Chess1\\16032100XrayNbBias\\', 'APA6_OUT5_Nb_700_0.txt')


#Analysis of data 24/03/16
#WaveformAnalysisV3('W:\\hvcmos01\\data\\Chess1\\16031100XraysFluorescence\\', 'APA1_OUT5_Nb0_0.txt')
#WaveformAnalysisV3('W:\\hvcmos01\\data\\Chess1\\16031100XraysFluorescence\\', 'APA1_OUT5_Sn0_0.txt')

#WaveformAnalysisV3('W:\\hvcmos01\\data\\Chess1\\16030700XraysFluorescence\\', 'APA4_OUT5_Nb0_0.txt')
#WaveformAnalysisV3('W:\\hvcmos01\\data\\Chess1\\16030700XraysFluorescence\\', 'APA4_OUT5_Sn0_0.txt')
#WaveformAnalysisV3('W:\\hvcmos01\\data\\Chess1\\16030700XraysFluorescence\\', 'APA7_OUT5_Cu0_0.txt')
#WaveformAnalysisV3('W:\\hvcmos01\\data\\Chess1\\16030700XraysFluorescence\\', 'APA7_OUT5_Fe0_0.txt')
#WaveformAnalysisV3('W:\\hvcmos01\\data\\Chess1\\16030700XraysFluorescence\\', 'APA7_OUT5_Nb0_0.txt')
#WaveformAnalysisV3('W:\\hvcmos01\\data\\Chess1\\16030700XraysFluorescence\\', 'APA7_OUT5_Sn0_0.txt')
#WaveformAnalysisV3('W:\\hvcmos01\\data\\Chess1\\16030700XraysFluorescence\\', 'APA8_OUT5_Cu0_0.txt')
#WaveformAnalysisV3('W:\\hvcmos01\\data\\Chess1\\16030700XraysFluorescence\\', 'APA8_OUT5_Fe0_0.txt')
#WaveformAnalysisV3('W:\\hvcmos01\\data\\Chess1\\16030700XraysFluorescence\\', 'APA8_OUT5_Nb0_0.txt')


#for file in os.listdir('W:\\hvcmos01\\data\\Chess1\\16030400XraysFluorescence\\'):
#    if file.endswith(".txt"):
#        WaveformAnalysisV3('W:\\hvcmos01\\data\\Chess1\\16030400XraysFluorescence\\', file)

#for file in os.listdir('W:\\hvcmos01\\data\\Chess1\\16030300XraysFluorescence\\'):
#    if file.endswith(".txt"):
#        WaveformAnalysisV3('W:\\hvcmos01\\data\\Chess1\\16030300XraysFluorescence\\', file)

#for file in os.listdir('W:\\hvcmos01\\data\\Chess1\\16030200XraysFluorescence\\'):
#    if file.endswith(".txt"):
#        WaveformAnalysisV3('W:\\hvcmos01\\data\\Chess1\\16030200XraysFluorescence\\', file)

#for file in os.listdir('W:\\hvcmos01\\data\\Chess1\\16030100XraysFluorescence\\'):
#    if file.endswith(".txt"):
#        WaveformAnalysisV3('W:\\hvcmos01\\data\\Chess1\\16030100XraysFluorescence\\', file)


#start_time = time.time()
#WaveformAnalysisV3('D:\\data\\Chess1\\16041400DACsPlayground\\', 'APA6_OUT5_Nb_DACS_00_40V0_0.txt')
#end_time = time.time()
#print("Processing time %g seconds" % ((end_time - start_time))) 

#PlotWaveforms('D:\\data\\Chess1\\16040800APA6NbUniformity\\', 'APA4_OUT1_Nb0_0.txt')

#obj = os.listdir('D:\\data\\Chess1\\16040801XrayNbBiasAPA4\\')
#for i in obj:
#    WriteTiff('D:\\data\\Chess1\\16040801XrayNbBiasAPA4\\', i)


#obj = os.listdir('W:\\hvcmos01\\data\\LFoundryA\\16090800NbVPFB\\')
#for _file in obj:
#    if _file.endswith(".txt"):
    #if "[20_50]Sn_Bias-20V" in _file:
#        WriteTiff('W:\\hvcmos01\\data\\LFoundryA\\16090800NbVPFB\\', _file)
 #       WaveformAnalysisV3('W:\\hvcmos01\\data\\LFoundryA\\16090800NbVPFB\\', _file)
   

#basepath = 'W:\\hvcmos01\\data\\LFoundryA\\16051900XRaysNb\\'
#for fname in os.listdir(basepath):
#    path = os.path.join(basepath, fname)
#    if os.path.isdir(path):
#        obj = os.listdir(path)
#        for _file in obj:
#            if _file.endswith(".txt"):
#                WaveformAnalysisV3(path + '\\', _file)
#                WriteTiff(path + '\\', _file)


#print os.path.isdir(obj[21])

#plt.show()

# def run(i):
#     time.sleep(2)
#     print('ok ' + str(i))
# 
# from threading import Thread
# for i in range(10):

#     t = Thread(target=run, args=(i,))
#     t.start()

        
#for i in range(10):
    ##self.GetWaveform(i)
    ##self.peakHisto.append(self.GetMaximum())
#    try:
#        thread.start_new_thread(MT, (i))
#    except:
#        print "Error: unable to start thread " + str(i)
