'''
Created on 20 Dec 2018

@author: Dima Maneuski
@author: Neil Moffat
@version: 6.0
@change: 1 - implemented new rise time extraction mechanism. Now the rise time is extracted between 3x noise (as in V3) and 90% of the peak position.
@change: 2 - Luis' playground for TowerJazz investigator
@change: 3 - assorted improvements to make graphs look better. In particular implemented style to plt.
@version: 7.0
@change: improvements in displaying results
@version: 8.0
@change: added new way of extracting peak position for TJ device. Now instead of extraction the peak of the waveform, it fits flat line to them the majour flat region is. 
@version: 8.1
@change: Branching from 8.0. Fixed the bugs that prevented to use V8 on LF chip waveforms.
@change: adding in method to extract timing information
@version: 1.1
'''

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pylab as py
import numpy as np
import time
import os
import glob
import tifffile
from tifffile import TiffWriter
from math import sqrt
import thread
from time import sleep

#plt.style.use('seaborn-notebook')
#http://matplotlib.org/users/customizing.html
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15

class WaveformAnalysis:

    
    def Init(self):
        self.const_frac = 0.1
        #self.const_frac_array = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        self.const_frac_array = [1.0]
        self.time_stamp_array = [[] for _ in range(len(self.const_frac_array))]
        self.lost_data =[]
        self.peak = 0
        self.x_at_min_bin = 0
        self.yData = 0
        self.x_at_ymin = 0
        self.xTHH_array = []
        self.yDataSet = []
        self.yDataSet_limits = []
        self.yDataSettime = []
        self.yDataSettime_limits = []
        self.yDataSetNegative = []
        self.xData = 0
        self.nPoints = 0
        self.rms_mean = []
        self.snr = []
        self.yMax = 0
        self.yMaxTJ = 0
        self.yTHH = 0
        self.xMax = 0
        self.xMin = 0
        self.xMinThreshold = 0
        self.yTHL = 0
        self.xTHL = 0
        self.xTHL_fit_time = 0
        self.xTHH_fit_time = 0
        self.fit_risetime = 0
        self.yTHLPercent = 0
        self.yTHLPeakMax = 0.1
        self.yTHLPeakMin = 0.5
        self.yTHLSigma = 3
        self.nWaveforms = 0
        self.iWaveform = 0
        self.deltaT = 0.0
        self.deltaTtime = 0.0
        self.baselineMean = 0.0
        self.baselineStd = 0.0
        self.timeBase = 5e-11
        self.timeCorrection = 5e-8
        self.noiseRange = 900
        self.wfQuality = 0
        self.risetime = 0
        self.filePath = ''
        self.fileName = ''
        self.fullFilePath = ''
        self.filePathtime = ''
        self.fileNametime = 'All_C3--trace_-10C_180V_all_timeforms_ordered.txt'
        self.fullFilePathtime = ''    
        self.peakHisto = []
        self.riseTimeHisto = []
        self.timestamp_10 = []
        
        self.baselineMeanHisto = []
        self.baselineStdHisto = []
        self.xData_fit_limits = []
        self.yData_fit_limits = []
        self.gaus_start = -2e-8
        self.gaus_end = 0
    def LoadFromFile(self, filePath, fileName):
        start_time = time.time()
        self.filePath = filePath
        self.fileName = fileName
        self.fullFilePath = self.filePath + self.fileName
        print("Opening file: %s" % (self.fullFilePath))
        f = open(self.fullFilePath, 'r')
        waveforms = f.readlines()[0:]
        counter = 0
        for i in waveforms:
            #print counter
            #counter = counter + 1
           # self.yDataSetNegative.append(np.asarray(i.split(','), float))
           self.yDataSet.append(np.asarray(i.split(','), float))
            
        #for i in self.yDataSetNegative:
         #   self.yDataSet.append(i*-1) 
          
        #if (self.yDataSet[0][0] < 1e-7): # This creates  back compatability with data taken before detlaT was written into file.
        self.deltaT = self.yDataSet[0][0]
        self.nWaveforms = len(waveforms)
        end_time = time.time()
        print("%d waveforms were opened in %g seconds" % (self.nWaveforms, (end_time - start_time)))
        # print self.yDataSet[0][0]
    '''Load a file which contains the timing information for the waveforms as the timebase does not seem to work for this, data taken from CERN'''
    def LoadFromFileTime(self, filePath):
        start_time = time.time()
        self.filePath = filePath
        #self.fileNametime = fileNametime
        self.fullFilePathtime = self.filePath + self.fileNametime
        print("Opening file: %s" % (self.fullFilePathtime))
        f = open(self.fullFilePathtime, 'r')
        timeforms = f.readlines()[0:]
        counter = 0
        for i in timeforms:
            #print counter
            #counter = counter + 1
           # self.yDataSetNegative.append(np.asarray(i.split(','), float))
           b = 0
           self.yDataSettime.append(np.asarray(i.split(','), float))
           

                   #self.yData_fit_limits.append(i)
           
             #  self.yDataSettime_limits.append(np.asarray(i.split(','), float))
        #for i in self.yDataSetNegative:
         #   self.yDataSet.append(i*-1) 
        #print self.yData_fit_limits
        #if (self.yDataSet[0][0] < 1e-7): # This creates  back compatability with data taken before detlaT was written into file.
        self.deltaTtime = self.yDataSettime[0][0]
        self.ntimeforms = len(timeforms)
        end_time = time.time()
        print("%d waveforms were opened in %g seconds" % (self.ntimeforms, (end_time - start_time)))
##############################################################################################
# self.yData - Y data of the waveform (amplitude)
# self.xData - X data of the waveform  (time converted to self.timeBase)
# self.nPoints - number of points in the waveform
# self.yMax - maximum amplitude of the waveform (along Y axis)
# self.yMin - minimum amplitude of the waveform (along Y axis)    
# self.xMax - x axis value where yMax occurs
# self.xMin - x axis value where yMax occurs
# self.yTHH - amplitude value of the waveform where yTHLPeakMax (default 90%) occurs
# self.xTHH - x axis value where yTHH occurs
# self.yTHL - 
# self.xTHL - x axis value where yTHL occurs
# self.yMaxTJ - version 8 of the peak position for the TJ devices 
# self.wfQuality - ranking the waveform for inclusion or exclusion from the analysis. Ranges from 0 to 10. 0 - exclude, 10 - include  
##############################################################################################
    def GetWaveform(self, iWaveform, value):
        if iWaveform > self.nWaveforms - 1:
            iWaveform = self.nWaveforms - 1
        
        self.iWaveform = iWaveform

        self.yData = self.yDataSet[self.iWaveform]
        self.nPoints = len(self.yData)
        #self.xData = np.asarray(np.linspace(0, self.nPoints*self.deltaT*self.timeBase, self.nPoints), float)
        self.xData = self.yDataSettime[self.iWaveform]
 

        p = 0
        for i in self.xData:
            p += 1
            if i > -2e-8 and i < 2e-8:
                self.xData_fit_limits.append(i)
                
                self.yData_fit_limits.append(self.yData[p])                             
           
            
         
        '''while self.gaus_start < 2e-8:
            self.xData_fit_limits.append(self.gaus_start)
            self.gaus_start = self.gaus_start + self.timeBase
            #print self.gaus_start
        self.gaus_end = self.gaus_start    
  '''

                #self.yData_fit_limits = self.yDataSet_limits[self.iWaveform]
        #self.xData_fit_limits = self.yDataSettime_limits[self.iWaveform]
        self.yMax = max(self.yData).astype(float)
        self.xMax = np.where(self.yData == self.yMax)[0][0].astype(float) * (self.deltaT*self.timeBase)
        
        self.yMin = min(self.yData).astype(float)
        
        self.xMin = np.where(self.yData == self.yMin)[0][0].astype(float) * (self.deltaT*self.timeBase)
        
        self.baselineMean = np.mean(self.yData[1:self.noiseRange]).astype(float)
        self.baselineStd = np.std(self.yData[1:self.noiseRange]).astype(float)

        summation = []
        for i in self.yData[0:self.noiseRange]:
            summation.append((i-self.baselineMean)**2)
        
        summation_2 = sum(summation)/self.noiseRange
        rms = sqrt(summation_2)
        
        self.rms_mean.append(rms)
        # TJ peak position extraction
        self.yMaxTJ = np.mean(self.yData[(len(self.yData) - self.noiseRange):(len(self.yData) - 1)]).astype(float)
        
        #print self.baselineMean
        # V6
#        self.yTHH = self.yData[np.where(self.yData > self.yTHLPeakMax*self.yMax)[0][0]]
#        self.xTHH = np.where(self.yData == self.yTHH)[0][0].astype(float) * (self.deltaT*self.timeBase)
        
#        self.yTHL = ((self.yTHH - self.yMin)*self.yTHLPercent/100).astype(float)
#        self.xTHL = self.xTHLQualifierV3()
        self.yMin = (self.yMin - self.baselineMean)
        self.snr.append(self.yMin/rms)
        print self.snr
        # V6_1
        #Need yTHH line but for th value of peak i need for any given value
        '''self.yTHH = self.yData[np.where(self.yData < (self.baselineMean + (self.yMin - self.baselineMean) ))[0][0]]
        
        #* (self.deltaT*self.timeBase)
        #self.xTHH = self.xTHH/self.timeBase
        self.x_at_ymin = np.where(self.yData == self.yMin)[0][0].astype(float) * self.timeBase - self.timeCorrection#* (self.deltaT*self.timeBase)
        self.x_at_min_bin = np.where(self.yData == self.yMin)[0][0].astype(float)
        self.x_at_min_bin = int(self.x_at_min_bin)
        
        self.xTHH = np.where(self.yData[:self.x_at_min_bin] == self.yTHH)[0][0].astype(float) * self.timeBase - self.timeCorrection
        self.xTHH_array = np.where(self.yData[:self.x_at_min_bin] > self.yTHH)#[0][0].astype(float) * self.timeBase - self.timeCorrection

        self.xTHH = max(self.xTHH_array[0]).astype(float) * self.timeBase - self.timeCorrection

        self.yTHL = self.yData[np.where(self.yData < (self.baselineMean + (self.yMin - self.baselineMean) * self.yTHLPeakMin))[0][0]] 
        self.xTHL = np.where(self.yData == self.yTHL)[0][0].astype(float) * self.timeBase - self.timeCorrection# * (self.deltaT*self.timeBase)
        #self.xTHL = self.xTHL/self.timeBase
        self.timestamp_10.append(self.xTHL)
        self.risetime = self.xTHH - self.xTHL'''

        
        #self.yTHL = (self.baselineMean - self.yTHLSigma*self.baselineStd).astype(float)
        #self.xTHL = self.xTHLQualifierV3()
        
        #self.wfQuality = self.wfQualityQualifierTJ()
        self.wfQuality = self.wfQualityQualifierLF()

        
    def xTHLQualifierV3(self):
        #waveform data between 0 and peak maximum
        dataY = self.yData[0:np.where(self.yData == self.yTHH)[0][0].astype(int)]
        dataY = np.atleast_2d(dataY)
        dataY = np.fliplr(dataY)
        dataY = dataY[0] #waveform data between 0 and peak maximum reversed (starting with peak maximum)
        dataYPosAboveNoise = np.where(dataY > (self.baselineMean + self.yTHLSigma*self.baselineStd))
        ret = self.xTHH - len(dataYPosAboveNoise[0])*(self.deltaT*self.timeBase) 
        return ret
    
    def wfQualityQualifierTJ(self):
        ret = 10 # ranges from 0 (exclude) to 10 (include)
        
        if self.yMaxTJ < (self.baselineMean + self.yTHLSigma*self.baselineStd):
            ret = 0
        return ret
 
    def wfQualityQualifierLF(self):
        ret = 10 # ranges from 0 (exclude) to 10 (include)
        return ret 
    
    def xMinThresholdQualifier(self):
        ret = np.where(self.yData > self.yTHL)[0].astype(float) * (self.deltaT*self.timeBase)
        return ret[0]

    def xMinThresholdQualifierV2(self):
        data = self.yData[0:np.where(self.yData == self.yTHH)[0][0].astype(float)]
        data = np.atleast_2d(data)
        data = np.fliplr(data)
        data = data[0]
        bb = np.where(data > self.yTHL)
        ret = bb[len(bb)-1].astype(float)
        ret = len(self.xData[0:self.xMax]) - ret
        ret = ret * (self.deltaT*self.timeBase)
        ret = self.xMax - ret[0]
        return ret
    
    def PlotWaveform(self, iWaveform, value):
        self.GetWaveform(iWaveform, value)
        fig = plt.figure()
        
        
        plt.scatter(self.xData, self.yData, marker="o", s=10, color="black", label="Waveform " + str(self.iWaveform))
        plt.axis([-5e-8,5e-8,-0.03, 0.03])

        #popt, pcov = curve_fit(Gaussian, self.xData, self.yData)

        sigma_start = 1e-9
        plt.ylabel("Voltage [V]")
        
        threshold = 1
        fit = 'gaus'
        '''
        try:
            if (fit == 'gaus'):
            
                #count = counts[threshold:]
            
                #bin = bins[threshold:]
                
                peakposition = np.where(self.yData == np.amin(self.yData))[0][0]
                print peakposition
                init_vals = [np.amin(self.yData),self.xData[peakposition] , sigma_start, self.baselineMean]
                
                
                popt, pcov = curve_fit(Gaussian, self.xData, self.yData, p0=init_vals)
                print popt

                
                #print "init vals: " + str(init_vals)
            ## second fitting iteration...
            #init_vals = popt
            #binSize = bins[1] - bins[0]
            
                peakIndex = np.where(self.yData > popt[1])[0][0]

                sigma1IndexLeft = np.where(self.yData < (popt[0]-1*popt[2]))[0][0]
                #sigma2IndexLeft = np.where(self.yData < (popt[1]-2*popt[2]))[0][0]        
                #sigma3IndexLeft = np.where(self.yData > (popt[1]-3*popt[2]))[0][0]
                sigma1IndexRight = np.where(self.yData > (popt[0]+1*popt[2]))[0][0]
                #sigma2IndexRight = np.where(self.yData > (popt[1]+2*popt[2]))[0][0]
                #sigma3IndexRight = np.where(self.yData > (popt[1]+3*popt[2]))[0][0]

                #sigmaIndexLeft = np.argmax(count) - 3
                #sigmaIndexRight = np.argmax(count) + 10
  
                sigma3IndexLeft = sigma1IndexLeft - 200
                sigma3IndexRight = sigma1IndexLeft + 100
            #delta = peakIndex - sigmaIndex
            
            #print peakIndex
            #print sigmaIndex
    
            #popt, pcov = curve_fit(Gaussian, [sigmaIndexLeft:len(bins)-1], counts[sigmaIndexLeft:len(bins)-1], p0=init_vals)
                popt, pcov = curve_fit(Gaussian, self.xData[sigma3IndexLeft:sigma3IndexRight], self.yData[sigma3IndexLeft:sigma3IndexRight], p0=init_vals)
                plt.plot(self.xData[sigma3IndexLeft:sigma3IndexRight],Gaussian(self.xData[sigma3IndexLeft:sigma3IndexRight],*popt), 'ro:',label='fit')
                #popt [0] = np.amax(count)# 2 * popt [0]
            #print popt
                max = min(Gaussian(self.xData[sigma3IndexLeft:sigma3IndexRight], *popt))
    
                max_bin = np.where(Gaussian(self.xData[sigma3IndexLeft:sigma3IndexRight], *popt) == max)
                
                fraction = max*self.const_frac_array[value]

                
                half_max_bin_low = np.where(Gaussian(self.xData[sigma3IndexLeft:sigma3IndexRight], *popt) < fraction)
                
                
                
                fwhm_array = []
                for i in half_max_bin_low[0]:
                    fwhm_array.append(i)
                    
                time_stamp_y = min(fwhm_array)
                time_stamp = self.xData[min(fwhm_array)+sigma3IndexLeft]
                self.time_stamp_array[value].append(time_stamp)
                print time_stamp
                #for i in half_max_bin_low[0]:
                 #   fwhm_array.append(i)
                    
                #time_stamp_y = min(fwhm_array)
                #time_stamp = self.xData[min(fwhm_array)+sigma3IndexLeft]
                #self.time_stamp_array.append(time_stamp)
                
    #       plt.plot(bins[sigmaIndex:len(bins)-1],Gaussian(bins[sigmaIndex:len(bins)-1],*popt),'ro:',label='fit')
                #plt.plot(self.xData_fit_limits,Gaussian(self.xData_fit_limits,*popt), 'ro:',label='fit', color='blue')
    
            
               # _res = abs(2.355 * 100 * round((popt[2]), 4) / round((popt[1]), 4));
                txt = "$\mu = $" + str(round((popt[1]), 11)) + "$\pm$" + str(round((popt[2]), 11)) + " [V]\n"
                plt.annotate(txt, xy=(0.05, 0.90), xycoords='axes fraction')    
                #fit_min = np.where(popt[0])[0][0]
                fit_array = Gaussian(self.xData[sigma3IndexLeft:sigma3IndexRight],*popt)
                fit_arrayx = self.xData[sigma3IndexLeft:sigma3IndexRight]
                fit_max = min(fit_array)

                
                #yfit_min = fit_array[np.where(fit_array > self.yTHLPeakMax*fit_array)]
                yTHH_fit = fit_array[np.where(fit_array < (self.baselineMean + (fit_max - self.baselineMean) * self.yTHLPeakMax))[0][0]]
                xTHH_fit = np.where(fit_array == yTHH_fit)[0][0].astype(float) #* self.timeBase - self.timeCorrection#* (self.deltaT*self.timeBase
                yTHL_fit = fit_array[np.where(fit_array < (self.baselineMean + (fit_max - self.baselineMean) * self.yTHLPeakMin))[0][0]]
                xTHL_fit = np.where(fit_array == yTHL_fit)[0][0].astype(float) #* self.timeBase - self.timeCorrection#* (self.deltaT*self.timeBase
                
               
                
                
                self.xTHL_fit_time = fit_arrayx[int(xTHL_fit)]
                self.xTHH_fit_time = fit_arrayx[int(xTHH_fit)]
                self.fit_risetime = self.xTHH_fit_time - self.xTHL_fit_time
                plt.axvline(time_stamp, 0.1, 0.9)
                plt.axvline(self.xTHL_fit_time, 0.1, 0.9)
                plt.axhline(self.yMin, 0.1, 0.9, color='red', linestyle='dashed')
                plt.axhline(self.yTHH, 0.1, 0.9, color='green', linestyle='dashed')
                  
        except:
            print "Sorry. Gaussian fit is not happening boyo....." 
            self.time_stamp_array[value].append(10)
            self.lost_data.append(1)
        '''
                

        #self.peak_value.append(self.yTHH)
        self.time_stamp_array[value].append(self.yMin)
        print len(self.time_stamp_array[value]), value
        plt.xlabel("Time [ns]")
        plt.grid(True)
        plt.legend()
        #plt.axvline(self.xTHL, 0.1, 0.9)
        #plt.axvline(self.xTHH, 0.1, 0.9)
     
        #plt.axvline(self.xData[self.noiseRange], 0.1, 0.2, color='green')
        '''plt.axhline(self.baselineMean, 0.1, 0.9, color='purple')
        plt.axhline(self.yTHL, 0.1, 0.9, color='blue', linestyle='dashed')
        plt.axhline(self.yMin, 0.1, 0.9, color='red', linestyle='dashed')
        plt.axhline(self.yTHH, 0.1, 0.9, color='blue', linestyle='dashed')
        plt.axvline(self.xTHH, 0.1, 0.9, color='blue', linestyle='dashed')
        #plt.axhline(self.yMax, 0.1, 0.9, color='red')
        #plt.axhline(self.yMin, 0.1, 0.9, color='red')
        #popt, pcov = curve_fit(Gaussian, self.xData, self.yData)
        #plt.axhline(self.yMaxTJ, 0.1, 0.9, color='red')
        #plt.plot(self.xData, Gaussian(self.xData,*popt), color="red")
        #plt.plot(bins[sigmaIndexLeft:sigmaIndexRight],Gaussian(bins[sigmaIndexLeft:sigmaIndexRight],*popt), 'ro:',label='fit', color='blue')
        txt = \
            'wfQuality = ' + str(round(self.wfQuality, 0)) + '\n' \
            'xTHH = ' + str(round(self.xTHH, 15)) + '\n' \
            'xTHL = ' + str(round(self.xTHL, 15)) + '\n' \
            'yTHH = ' + str(round(self.yTHH, 15)) + '\n' \
            'yTHL = ' + str(round(self.yTHL, 15)) + '\n' 
            #'10% value = ' + str(round(time_stamp, 15)) + '\n' \
            #'yMax = ' + str(round(self.yMax, 15)) + '\n' 
            #'yMaxTJ = ' + str(round(self.yMaxTJ, 4)) + '\n' \
            #'yMin = ' + str(round(self.yMin, 4)) + '\n' \
#            'baselineStd = ' + str(round(self.baselineStd, 4)) + '\n' \
            #'baselineMean = ' + str(round(self.baselineMean, 4)) + '\n' \
 #           '-----------------------------------------' + '\n' \
  #          'xTHH - xTHL = ' + str(round(self.xTHH - self.xTHL, 15)) + '\n' \
   #         'xTHH_fit - xTHL_fit = ' + str(round(self.xTHH_fit_time - self.xTHL_fit_time, 15)) + '\n' \
    #        'yMax - baselineMean = ' + str(round(self.yMax - self.baselineMean, 4)) + '\n'

        plt.annotate(txt, xy=(0.60, 0.02), xycoords='axes fraction', fontweight='regular')    '''
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #plt.show()
        plt.close(fig)
        
        return data

    def GetMaximum(self):
        return self.yMax
        
    def GetRiseTime(self):
        
        
        return self.fit_risetime
       # return self.xTHH - self.xTHL

    def GetBaselineMean(self):
        return self.baselineMean


    def CalculateHistograms(self):
        counter = []
        for i in range(self.nWaveforms):
            self.GetWaveform(i)
            if (self.wfQuality >= 5.0):
                self.peakHisto.append(self.yMin - self.GetBaselineMean())
                self.riseTimeHisto.append(self.GetRiseTime())
                self.baselineMeanHisto.append(self.baselineMean)
                self.baselineStdHisto.append(self.baselineStd)
            else:
                # print "Histogram %d quality is %d" %(i, self.wfQuality)
                counter.append(i)

        print "# histograms of quality 0 is %d" %(len(counter))
        print counter        
               
# Changes in V6  
# instead of getting just max postion we take the difference between Noise level and Peak position 
#    
    def GetPeakHistogram(self):
#         for i in range(self.nWaveforms):
#             self.GetWaveform(i)
#             # self.peakHisto.append(self.GetMaximum() - self.GetBaselineMean())
#             # see version 8
# 
#             if (self.wfQuality > 5.0):
#                 self.peakHisto.append(self.yMaxTJ - self.GetBaselineMean())
#             else:
#                 print "Histogram %d quality is %d" %(i, self.wfQuality)
#                 self.peakHisto.append(-1.0)            
            
        return self.peakHisto
    
    def GetRiseTimeHistogram(self):
#         for i in range(self.nWaveforms):
#             self.GetWaveform(i)
#             self.riseTimeHisto.append(self.GetRiseTime())

        return self.riseTimeHisto
    
    def GetBaselineMeanHistogram(self):
#         for i in range(self.nWaveforms):
#             self.GetWaveform(i)
#             if (self.wfQuality > 5.0):
#                 self.baselineMeanHisto.append(self.baselineMean)
#             else:
#                 print "Histogram %d quality is %d" %(i, self.wfQuality)
#                 self.baselineMeanHisto.append(-1.0)
        return self.baselineMeanHisto

    def GetBaselineStdHistogram(self):
#         for i in range(self.nWaveforms):
#             self.GetWaveform(i)
#             if (self.wfQuality > 5.0):
#                 self.baselineStdHisto.append(self.baselineStd)
#             else:
#                 print "Histogram %d quality is %d" %(i, self.wfQuality)
#                 self.baselineStdHisto.append(-1.0)  

        return self.baselineStdHisto
    
    def SaveWaveformsAsTiff(self, fullFilePath):
        self.yTHLPercent = 15
        for i in range(self.nWaveforms):
            self.GetWaveform(i)

    def MT(self, ttt):
        time.sleep(1)
        print "I am thread: " + str(ttt)
            ##self.GetWaveform(wfNumber)
            ##self.peakHisto.append(self.GetMaximum())

def WriteTiff(filePath, fileName):
    
    analysis = WaveformAnalysis()
    analysis.Init()
    analysis.LoadFromFile(filePath, fileName)
    analysis.LoadFromFileTime(filePath)
    #analysis.yTHLPercent = 15
    #data = analysis.PlotWaveform(19)
    #for i in range(analysis.nWaveforms):
    #    data = analysis.PlotWaveform(i)
        #plt.show()
        #plt.savefig("C:\\analysis.tif")
    
#     data = np.random.randint(0, 255, (256, 256)).astype(np.uint8)
#     data1 = np.random.randint(0, 255, (256, 256)).astype(np.uint8)
#     
#     for i in range(self.nWaveforms):
#         analysis.PlotWaveform(i)
#         plt.show()
#   
    filenameNoExt, file_extension = os.path.splitext(analysis.fileName)
    start_time = time.time()
    with TiffWriter(filePath + filenameNoExt + '.tif') as tif:
        for value in range(len(analysis.const_frac_array)):
            for i in range(analysis.nWaveforms): #analysis.nWaveforms
                data = analysis.PlotWaveform(i,value)
                tif.save(data, compress=6)
    end_time = time.time()
    print("%d waveforms were processed in %g seconds" % (analysis.nWaveforms, (end_time - start_time))) 
    with open(filePath+fileName+'time_stamp'+str(analysis.const_frac)+'.txt', 'w') as f:
        for item in analysis.time_stamp_array:
            f.write("%s\t" % item) 
    f.close()
    
def Time_resolution(filePath, fileName):
    
    analysis = WaveformAnalysis()
    analysis.Init()
    analysis.LoadFromFile(filePath, fileName)
    analysis.LoadFromFileTime(filePath)
    #analysis.yTHLPercent = 15
    #data = analysis.PlotWaveform(19)
    #for i in range(analysis.nWaveforms):
    #    data = analysis.PlotWaveform(i)
        #plt.show()
        #plt.savefig("C:\\analysis.tif")
    
#     data = np.random.randint(0, 255, (256, 256)).astype(np.uint8)
#     data1 = np.random.randint(0, 255, (256, 256)).astype(np.uint8)
#     
#     for i in range(self.nWaveforms):
#         analysis.PlotWaveform(i)
#         plt.show()
#   
    filenameNoExt, file_extension = os.path.splitext(analysis.fileName)
    start_time = time.time()
    #with TiffWriter(filePath + filenameNoExt + '.tif') as tif:
    for value in range(len(analysis.const_frac_array)):
        for i in range(analysis.nWaveforms): #analysis.nWaveforms
            data = analysis.PlotWaveform(i,value)
            #tif.save(data, compress=6)
    end_time = time.time()
    rms_final_mean =  sum(analysis.rms_mean)/len(analysis.rms_mean)
    min_noise = min(analysis.rms_mean)
    max_noise = max(analysis.rms_mean)
    snr_mean = sum(analysis.snr)/len(analysis.snr)
    print "final rms mean"
    print rms_final_mean
    print "final snr mean"
    print snr_mean
    print "Max Noise:  ", max_noise
    print "Min Noise:  ", min_noise
    #filename = filename.split('.')
    print("%d waveforms were processed in %g seconds" % (analysis.nWaveforms, (end_time - start_time))) 
    with open(filePath+filenameNoExt +'_MPV.txt', 'w') as f:
        for item in analysis.time_stamp_array:
            for i in item:
                f.write("%s\t" % i)
            f.write("\n") 
    f.close()    

def PlotWaveforms(filePath, fileName):
    analysis = WaveformAnalysis()
    analysis.Init()
    analysis.LoadFromFile(filePath, fileName)
    analysis.LoadFromFileTime(filePath)
    analysis.yTHLPercent = 15
    #analysis.PlotWaveform(18)
    for i in range(analysis.nWaveforms):
        analysis.PlotWaveform(i)
        plt.show()

def Gaussian(x, a, x0, sigma, offset):
    return a*py.exp(-(x-x0)**2/(2*sigma**2)) + offset

def WaveformAnalysis_Timing_V1(filePath, fileName, xLimitMinNoise = -0.01, xLimitMinSignal = 0, xLimitMinRiseTime = 0,
                        xLimitMaxNoise = 0.050, xLimitMaxSignal = 0.05, xLimitMaxRiseTime = 9e-9):
    analysis = WaveformAnalysis()
    analysis.Init()
    analysis.LoadFromFile(filePath, fileName)
    analysis.LoadFromFileTime(filePath)
    analysis.yTHLPercent = 15


    start_time = time.time()
    
    nBins = 100
#    _fontSize = 18

    analysis.CalculateHistograms()
    
    _outPath = filePath + 'outputV8_1' + '/'
    if not os.path.exists(_outPath):
        os.makedirs(_outPath)

    ######################## Baseline histogram #######################
    fig = plt.figure()
    plt.grid(True)
    counts, bins, ignored = plt.hist(analysis.GetBaselineMeanHistogram(), bins=nBins, normed=1, facecolor='purple', align='mid', histtype='stepfilled', alpha=1.0, label="Spectrum")
    #plt.title('Baseline level spectrum')
    plt.xlabel("Voltage [V]")
    plt.ylabel("Counts")
    plt.annotate(analysis.fullFilePath, xy=(0.01, 0.01), xycoords='axes fraction', color='black')
    #plt.xlim(analysis.baselineMean - 9*analysis.baselineStd, analysis.baselineMean + 9*analysis.baselineStd)
    filename, file_extension = os.path.splitext(analysis.fileName)
    
    fit = 'gaus'
     
    try:
        if (fit == 'gaus'):
            
    
            popt, pcov = curve_fit(Gaussian, bins[0:len(bins)-1], counts[0:len(bins)-1])
            #plt.plot(bins[0:len(bins)-1],Gaussian(bins[0:len(bins)-1],*popt),'ro:',label='fit', color='green')
#             init_vals = [np.amax(counts), bins[np.argmax(counts)], bins[np.argmax(counts)]/10]
#             x0 = 0
#             popt, pcov = curve_fit(Gaussian, bins[x0:len(bins)-1], counts[x0:len(bins)-1]) # ,p0=init_vals 
#             popt [0] = np.amax(counts) # 2 * popt [0]
#     
#             ## second fitting iteration...
#             #init_vals = popt
#             #binSize = bins[1] - bins[0]
#             
#             peakIndex = np.where(bins > popt[1])[0][0]
#             sigma1IndexLeft = np.where(bins > (popt[1]-1*popt[2]))[0][0]
#             sigma2IndexLeft = np.where(bins > (popt[1]-2*popt[2]))[0][0]        
            sigma3IndexLeft = np.where(bins > (popt[1]-3*popt[2]))[0][0]
#             sigma1IndexRight = np.where(bins > (popt[1]+1*popt[2]))[0][0]
#             sigma2IndexRight = np.where(bins > (popt[1]+2*popt[2]))[0][0]
            sigma3IndexRight = np.where(bins > (popt[1]+3*popt[2]))[0][0]
#     
            sigmaIndexRight = sigma3IndexRight
            sigmaIndexLeft = sigma3IndexLeft
            #delta = peakIndex - sigmaIndex
            
            #print peakIndex
            #print sigmaIndex
    
           #popt, pcov = curve_fit(Gaussian, bins[sigmaIndexLeft:len(bins)-1], counts[sigmaIndexLeft:len(bins)-1], p0=init_vals)
            
            #popt, pcov = curve_fit(Gaussian, bins[sigmaIndexLeft:sigmaIndexRight], counts[sigmaIndexLeft:sigmaIndexRight], p0=init_vals)
            
            #popt [0] = np.amax(counts)# 2 * popt [0]
#             print "Baseline params" + popt
            
            #plt.plot(bins[sigmaIndex:len(bins)-1],Gaussian(bins[sigmaIndex:len(bins)-1],*popt),'ro:',label='fit')
            plt.plot(bins[sigmaIndexLeft:sigmaIndexRight],Gaussian(bins[sigmaIndexLeft:sigmaIndexRight],*popt), 'ro:',label='fit', color='blue')

            
            txt = "$\mu = $" + str(round((popt[1]), 4)) + "$\pm$" + str(round((popt[2]), 4)) + " [V]"
            plt.annotate(txt, xy=(0.05, 0.90), xycoords='axes fraction')    
         
    except:
        print "Sorry. Baseline histogram ain't happening...."


    _outPath_ = _outPath + 'baseline' + '/'
    if not os.path.exists(_outPath_):
        os.makedirs(_outPath_)
        
    plt.savefig(_outPath_ + filename + '_baseline.png')
    plt.close(fig)


    ####################### Peak position histogram #######################
    fig = plt.figure()
    plt.grid(True)
    counts, bins, ignored = plt.hist(analysis.GetPeakHistogram(), bins=nBins, normed=1, facecolor='red', align='mid', histtype='stepfilled', alpha=1.0, label="Spectrum")
   # plt.title('Signal spectrum')
    plt.xlabel("Voltage [V]")
    plt.ylabel("Counts")
    plt.annotate(analysis.fullFilePath, xy=(0.01, 0.01), xycoords='axes fraction', color='black')
    plt.xlim(xLimitMinSignal, xLimitMaxSignal)
    filename, file_extension = os.path.splitext(analysis.fileName)
    
    fit = 'gaus'
    threshold = 10

    try:
        if (fit == 'gaus'):
            
            count = counts[threshold:]
            
            bin = bins[threshold:]
            
            init_vals = [np.amax(count), bin[np.argmax(count)], bin[np.argmax(count)]/10]
            x0 = 0
            popt, pcov = curve_fit(Gaussian, bin[x0:len(bin)-1], count[x0:len(bin)-1], p0=init_vals)
            popt [0] = np.amax(count) # 2 * popt [0]
            print "here"
            print "init vals: " + str(init_vals)
            ## second fitting iteration...
            #init_vals = popt
            #binSize = bins[1] - bins[0]
            
            #peakIndex = np.where(bins > popt[1])[0][0]
            #sigma1IndexLeft = np.where(bins > (popt[1]-1*popt[2]))[0][0]
            #sigma2IndexLeft = np.where(bins > (popt[1]-2*popt[2]))[0][0]        
            #sigma3IndexLeft = np.where(bins > (popt[1]-3*popt[2]))[0][0]
            #sigma1IndexRight = np.where(bins > (popt[1]+1*popt[2]))[0][0]
            #sigma2IndexRight = np.where(bins > (popt[1]+2*popt[2]))[0][0]
            #sigma3IndexRight = np.where(bins > (popt[1]+3*popt[2]))[0][0]
    
            sigmaIndexLeft = np.argmax(count) - 3
            sigmaIndexRight = np.argmax(count) + 10


            #delta = peakIndex - sigmaIndex
            
            #print peakIndex
            #print sigmaIndex
    
    #       popt, pcov = curve_fit(Gaussian, bins[sigmaIndexLeft:len(bins)-1], counts[sigmaIndexLeft:len(bins)-1], p0=init_vals)
            popt, pcov = curve_fit(Gaussian, bin[sigmaIndexLeft:sigmaIndexRight], count[sigmaIndexLeft:sigmaIndexRight], p0=init_vals)
            
            popt [0] = np.amax(count)# 2 * popt [0]
            #print popt
            
    #       plt.plot(bins[sigmaIndex:len(bins)-1],Gaussian(bins[sigmaIndex:len(bins)-1],*popt),'ro:',label='fit')
            plt.plot(bin[sigmaIndexLeft:sigmaIndexRight],Gaussian(bin[sigmaIndexLeft:sigmaIndexRight],*popt), 'ro:',label='fit', color='blue')
    
            
            _res = abs(2.355 * 100 * round((popt[2]), 4) / round((popt[1]), 4));
            txt = "$\mu = $" + str(round((popt[1]), 4)) + "$\pm$" + str(round((popt[2]), 4)) + " [V]\n" + "R = " + str(round(_res, 2)) + " %" 
            plt.annotate(txt, xy=(0.05, 0.90), xycoords='axes fraction')    
        
    except:
        print "Sorry. Peak position histogram ain't happening...."
    
    _outPath_ = _outPath + 'peak' + '/'
    if not os.path.exists(_outPath_):
        os.makedirs(_outPath_)
    
    
    plt.savefig(_outPath_ + filename + '_peak.png')
    plt.close(fig)
    
    ####################### Rise time histogram #######################
#   analysis.yTHLPercent = 25
    fig = plt.figure()
    plt.grid(True)
    try:
        counts, bins, ignored = plt.hist(analysis.GetRiseTimeHistogram(), bins=nBins, range=[xLimitMinRiseTime, xLimitMaxRiseTime], normed=1, facecolor='blue', align='mid', histtype='stepfilled', alpha=1.0, label="Rise time distribution")
    except:
        print "Sorry. Rise time histogram ain't happening...."    
    #plt.title('Rise time distribution')
    plt.xlabel("Rise time [ns]")
    plt.ylabel("Counts")
    plt.annotate(analysis.fullFilePath, xy=(0.01, 0.01), xycoords='axes fraction', color='black')
    plt.xlim(xLimitMinRiseTime, xLimitMaxRiseTime)
    filename, file_extension = os.path.splitext(analysis.fileName)
    
    fit = 'gaus'
    
    try:    
        if (fit == 'gaus'):
            init_vals = [np.amax(counts), bins[np.argmax(counts)], bins[np.argmax(counts)]/10]
            x0 = 0
            popt, pcov = curve_fit(Gaussian, bins[x0:len(bins)-1], counts[x0:len(bins)-1], p0=init_vals)
            popt [0] = np.amax(counts) # 2 * popt [0]
    
            ## second fitting iteration...
            #init_vals = popt
            #binSize = bins[1] - bins[0]
            
            peakIndex = np.where(bins > popt[1])[0][0]
            sigma3IndexLeft = np.where(bins > (popt[1]-3*popt[2]))[0][0]
            sigma1IndexRight = np.where(bins > (popt[1]+1*popt[2]))[0][0]
            sigma3IndexRight = np.where(bins > (popt[1]+3*popt[2]))[0][0]
    
            sigmaIndexRight = sigma3IndexRight
            #delta = peakIndex - sigmaIndex
            
            #print peakIndex
            #print sigmaIndex
    
    #       popt, pcov = curve_fit(Gaussian, bins[sigmaIndexLeft:len(bins)-1], counts[sigmaIndexLeft:len(bins)-1], p0=init_vals)
            
            popt, pcov = curve_fit(Gaussian, bins[sigma3IndexLeft:sigmaIndexRight], counts[sigma3IndexLeft:sigmaIndexRight], p0=init_vals)
        
            popt [0] = np.amax(counts)# 2 * popt [0]
            print popt
            
            #plt.plot(bins[sigmaIndex:len(bins)-1],Gaussian(bins[sigmaIndex:len(bins)-1],*popt),'ro:',label='fit')
            plt.plot(bins[sigma3IndexLeft:sigmaIndexRight],Gaussian(bins[sigma3IndexLeft:sigmaIndexRight],*popt), 'ro:', label='fit')
            txt = "$\mu = $" + str('{:.2e}'.format(popt[1])) + "$\pm$" + str('{:.2e}'.format(popt[2])) + " [ns]"
            #txt = "$\mu = $" + str(round((popt[1]), 9)) + "$\pm$" + str(round((popt[2]), 0)) + " [ns]"
            plt.annotate(txt, xy=(0.05, 0.90), xycoords='axes fraction')    
    except:
        print "Sorry. Rise time histogram ain't happening...."    
    
    
    _outPath_ = _outPath + 'rise' + '/'
    if not os.path.exists(_outPath_):
        os.makedirs(_outPath_)

    plt.savefig(_outPath_ + filename + '_rise.png')
    plt.close(fig)
    
    #######################  Rise time peak histogram #######################    
    fig = plt.figure()
    # Estimate the 2D histogram
    H, xedges, yedges = np.histogram2d(analysis.GetPeakHistogram(), analysis.GetRiseTimeHistogram(), bins=nBins)
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
    # Plot 2D histogram using pcolor
    plt.pcolormesh(xedges,yedges,Hmasked)
    plt.xlabel('Voltage [V]')
    plt.ylabel('Rise time [ns]')
    
    plt.xlim(xLimitMinSignal, xLimitMaxSignal)
    plt.ylim(xLimitMinRiseTime, xLimitMaxRiseTime)
    
    plt.annotate(analysis.fullFilePath, xy=(0.01, 0.01), xycoords='axes fraction', color='black')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.grid(True)
    filename, file_extension = os.path.splitext(analysis.fileName)
    
    _outPath_ = _outPath + 'peak_rise' + '/'
    if not os.path.exists(_outPath_):
        os.makedirs(_outPath_)

    plt.savefig(_outPath_ + filename + '_peak_rise.png')
    plt.close(fig)
    
    ####################### Noise histogram #######################
    fig = plt.figure()
    plt.grid(True)
    counts, bins, ignored = plt.hist(analysis.GetBaselineStdHistogram(), bins=nBins, range=[xLimitMinNoise, xLimitMaxNoise], normed=1, facecolor='green', align='mid', histtype='stepfilled', alpha=1.0, label="Noise histogram")
    #plt.title('Noise')
    plt.xlabel("Voltage [V]")
    plt.ylabel("Counts")

    plt.xlim(xLimitMinNoise, xLimitMaxNoise)
    plt.annotate(analysis.fullFilePath, xy=(0.01, 0.01), xycoords='axes fraction', color='black')
    filename, file_extension = os.path.splitext(analysis.fileName)
    
    fit = 'gaus'
    
    try: 
        if (fit == 'gaus'):
            init_vals = [np.amax(counts), bins[np.argmax(counts)], bins[np.argmax(counts)]/10]
            x0 = 0
            popt, pcov = curve_fit(Gaussian, bins[x0:len(bins)-1], counts[x0:len(bins)-1], p0=init_vals)
            popt [0] = np.amax(counts) # 2 * popt [0]
    
            ## second fitting iteration...
            #init_vals = popt
            #binSize = bins[1] - bins[0]
            
            peakIndex = np.where(bins > popt[1])[0][0]
            sigma3IndexLeft = np.where(bins > (popt[1]-3*popt[2]))[0][0]
            sigma1IndexRight = np.where(bins > (popt[1]+1*popt[2]))[0][0]
            sigma3IndexRight = np.where(bins > (popt[1]+3*popt[2]))[0][0]
    
            #delta = peakIndex - sigmaIndex
            
            #print peakIndex
            #print sigmaIndex
    
    #       popt, pcov = curve_fit(Gaussian, bins[sigmaIndexLeft:len(bins)-1], counts[sigmaIndexLeft:len(bins)-1], p0=init_vals)
            popt, pcov = curve_fit(Gaussian, bins[sigma3IndexLeft:sigma1IndexRight], counts[sigma3IndexLeft:sigma1IndexRight], p0=init_vals)
            
            popt [0] = np.amax(counts)# 2 * popt [0]
            print popt
            
    #       plt.plot(bins[sigmaIndex:len(bins)-1],Gaussian(bins[sigmaIndex:len(bins)-1],*popt),'ro:',label='fit')
            plt.plot(bins[sigma3IndexLeft:sigma1IndexRight],Gaussian(bins[sigma3IndexLeft:sigma1IndexRight],*popt), 'ro:',label='fit')
    
            
            txt = "$\mu = $" + str(round((popt[1]), 5)) + "$\pm$" + str(round((popt[2]), 5)) + " [V]"
            plt.annotate(txt, xy=(0.05, 0.90), xycoords='axes fraction')
    except:
        print "Sorry. Noise histogram ain't happening...."    

    _outPath_ = _outPath + 'noise' + '/'
    if not os.path.exists(_outPath_):
        os.makedirs(_outPath_)

    print "Saving noise plot to %s" % (_outPath_ + filename + '_noise.png')
    plt.savefig(_outPath_ + filename + '_noise.png')
    plt.close(fig)
    
    end_time = time.time()
    print("%d waveforms were processed in %g seconds" % (analysis.nWaveforms, (end_time - start_time))) 
    #WaveformAnalysisNoise(filePath, fileName)

    ####################### Signal, baseline and noise spectra combined #######################
    fig = plt.figure()
    plt.grid(True)
    plt.hist(analysis.GetPeakHistogram(), bins=nBins, normed=1, facecolor='red', align='mid', histtype='stepfilled', alpha=1.0, label="Spectrum")
    counts, bins, ignored = plt.hist(analysis.GetBaselineMeanHistogram(), bins=nBins, normed=1, facecolor='purple', histtype='stepfilled', alpha=1.0, label="Baseline")
    plt.hist(analysis.GetBaselineStdHistogram(), bins=nBins, normed=1, facecolor='green', histtype='stepfilled', alpha=1.0, label="Noise")
    #plt.title('Signal, baseline and noise spectra combined')
    plt.xlabel("Voltage [V]")
    plt.ylabel("Counts")
    plt.ylim((0,max(counts) + 0.2*max(counts)))
    
    _outPath_ = _outPath + 'noise_peak' + '/'
    if not os.path.exists(_outPath_):
        os.makedirs(_outPath_)

    plt.savefig(_outPath_ + filename + '_noise_peak.png')
    plt.close(fig)



    print "This the timestamp"
    print analysis.timestamp_10
    with open(filePath+filename+'time_stamp50.txt', 'w') as f:
        for item in analysis.timestamp_10:
            f.write("%s\n" % item) 
    f.close()

def WaveformAnalysisNoise(filePath, fileName):
    analysis = WaveformAnalysis()
    analysis.Init()
    analysis.LoadFromFile(filePath, fileName)
    analysis.yTHLPercent = 15
    #analysis.PlotWaveform(10)
    
#    analysis.GetNoise()
  
    nBins = 50
#      
#     plt.figure()
#     plt.grid(True)
#     counts, bins, ignored = plt.hist(analysis.GetBaselineMeanHistogram(), bins=nBins, normed=1, facecolor='blue', histtype='stepfilled', alpha=1.0, label="Frame histogram")
#     plt.title('Baseline mean')
#     plt.xlabel("Voltage [V]")
#     plt.ylabel("Counts")
#     plt.annotate(analysis.fullFilePath, xy=(0.01, 0.01), xycoords='axes fraction', color='black')
#     filename, file_extension = os.path.splitext(analysis.fileName)
#     plt.savefig(analysis.filePath + 'output\\' + filename + '_BLMean_th=' + str(analysis.yTHLPercent) + '.png')
#  
#     #plt.xlim(0.0, 0.20)
#  
    plt.figure()
    plt.grid(True)
    counts, bins, ignored = plt.hist(analysis.GetBaselineStdHistogram(), bins=nBins, normed=1, facecolor='red', histtype='stepfilled', alpha=1.0, label="Frame histogram")
    #plt.title('Noise')
    plt.xlabel("Voltage [V]")
    plt.ylabel("Counts")
    plt.xlim(0.0, 0.005)
    plt.annotate(analysis.fullFilePath, xy=(0.01, 0.01), xycoords='axes fraction', color='black')
    filename, file_extension = os.path.splitext(analysis.fileName)
    plt.savefig(analysis.filePath + 'output/' + filename + '_Noise_th=' + str(analysis.yTHLPercent) + '.png')