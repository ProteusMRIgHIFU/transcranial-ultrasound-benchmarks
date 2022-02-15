'''
Library for ITRUSST intercomparison effort with BabelViscoFDTD
https://github.com/ProteusMRIgHIFU/BabelViscoFDTD

Samuel Pichardo, Ph.D
Assistant Professor
Radiology and Clinical Neurosciences, Hotchkiss Brain Institute
Cumming School of Medicine,
University of Calgary
samuel.pichardo@ucalgary.ca
www.neurofus.ca

'''
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from BabelViscoFDTD.H5pySimple import ReadFromH5py,SaveToH5py
from BabelViscoFDTD.PropagationModel import PropagationModel
from scipy import ndimage
from scipy import interpolate
from skimage.draw import circle_perimeter,disk
from skimage.transform import rotate
import mat73
import cupy 
import cupyx 
from cupyx.scipy import ndimage as cndimage
import time
from IPython.display import display,clear_output

import gc

import mkl_fft

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

############################
PModel=PropagationModel()

MatlabEng = None

## Global definitions

DbToNeper=1/(20*np.log10(np.exp(1)))

Material={}
#Density (kg/m3), LongSoS (m/s), ShearSoS (m/s), Long Att (Np/m), Shear Att (Np/m)
Material['Water']=     np.array([1000.0, 1500.0, 0.0,   0.0,                   0.0] )
Material['SofTissue']= np.array([1000.0, 1500.0, 0.0,   1.0 * DbToNeper * 100,  0.0] )
Material['Cortical']=  np.array([1850.0, 2800.0, 1550.0, 4.0 * DbToNeper * 100, 12 * DbToNeper * 100])
Material['Trabecular']=np.array([1700.0, 2300.0, 1400.0, 8.0 * DbToNeper * 100, 12 * DbToNeper * 100])
Material['Skin']=      np.array([1090.0, 1610.0, 0.0,    0.2 * DbToNeper * 100, 0])
Material['Brain']=     np.array([1040.0, 1560.0, 0.0,    0.3 * DbToNeper * 100, 0])

################################################################################################################
################################################################################################################
################################################################################################################

class ITRUSST_Simulations(object):
    '''
    Meta class dealing with the specificis of each test based on the string name
    '''
    
    def __init__(self,TESTNAME='',
                 RasterInputPath='.'+os.sep,
                 OutputPath='.'+os.sep,
                 bDisplay=True,
                 OverwriteCFL=None,
                 bSourceDisplacement=False):
        self._TESTNAME=TESTNAME
        lSplit=TESTNAME.replace('_','-').split('-')
        self._Phase=lSplit[0]
        self._Benchmark=lSplit[1]
        self._SubType=lSplit[2]
        self._SourceType=lSplit[3]
        self._SimType=lSplit[4]
        if self._Benchmark in ['BM2','BM3','BM4','BM5','BM6','BM7','BM8'] and self._SubType == 'MP1':
            self._Shear=0.0
        else:
            self._Shear=1.0
        
        if '24PPW' in lSplit:
            self._basePPW=24
        elif '18PPW' in lSplit:
            self._basePPW=18
        elif '15PPW' in lSplit:
            self._basePPW=15
        elif '12PPW' in lSplit:
            self._basePPW=12
        elif '9PPW' in lSplit:
            self._basePPW=9
        elif '6PPW' in lSplit:
            self._basePPW=6
        
        assert(self._basePPW  in [6,9,12,15,18,24])

        self._PaddingForKArray=15
        
        self._AlphaCFL=1.0
        self._bDisplay=bDisplay
        
        self._bSourceDisplacement=bSourceDisplacement
        
        self._RasterInputPath=RasterInputPath
        self._OutputPath=OutputPath
        
        
        if not(OverwriteCFL is None):
            self._AlphaCFL=OverwriteCFL
               
    
    def GetSummary(self):
        summary=''
        summary+='### Benchmark type: '
        if self._Benchmark=='BM1':
            summary+='Homogenous medium'
            if self._SubType == 'MP1':
                summary+=' - Lossless'
            else:
                summary+=' - Absorbing'
        elif self._Benchmark=='BM2':
            summary+='Single flat cortical layer'
        elif self._Benchmark=='BM3':
            summary+='Composite flat layer'
        elif self._Benchmark=='BM4':
            summary+='Single curved cortical layer'
        elif self._Benchmark=='BM5':
            summary+='Composite curved cortical layer'
        elif self._Benchmark=='BM6':
            summary+='Half skull (homogenous) targeting V1'
        elif self._Benchmark=='BM7':
            summary+='Full skull (homogenous) targeting V1'
        elif self._Benchmark=='BM8':
            summary+='Full skull (homogenous) targeting M1'
            
        if self._Benchmark != 'BM1':
            if self._SubType == 'MP1':
                summary+='\n### Acoustic conditions (shear component disabled in elastic model)'
            else:
                summary+='\n### Elastic conditions'
        if self._SourceType =='SC1':
            summary+='\n### Bowl source'
        else:
            summary+='\n### Piston source'
            
        summary+='\n### Resolution: %i PPW' %(self._basePPW)
        
        return summary


    def Step1_InitializeConditions(self):
        
        NumberCyclesToTrackAtEnd=2
        if self._Benchmark=='BM1':
            if self._SubType=='MP1':
                #the PPP below depend of several factors as the lowest speed of sound and PPW, the subsamplig (by the time being) is adjusted case by case
                BaseSupSamplig={6:1,9:1,12:5,15:1,18:1,24:7}
            else:
                BaseSupSamplig={6:1,9:4,12:3,15:3,18:4,24:1}
        else:
            if self._SubType=='MP1' or self._Benchmark in ['BM2','BM4','BM6','BM7','BM8']:
                BaseSupSamplig={6:4,9:9,12:8,15:10,18:9,24:19}
            else:
                BaseSupSamplig={6:2,9:3,12:3,15:8,18:11,24:6}
        SensorSubSampling=BaseSupSamplig[self._basePPW]
       
        if self._Benchmark=='BM1' and self._SubType=='MP2':
            self._SIM_SETTINGS = SimulationConditions(baseMaterial=Material['SofTissue'],
                                basePPW=self._basePPW,
                                SensorSubSampling=SensorSubSampling,  
                                NumberCyclesToTrackAtEnd=NumberCyclesToTrackAtEnd,                      
                                PaddingForKArray=self._PaddingForKArray,
                                bDisplay=self._bDisplay,
                                DispersionCorrection=[-2285.16917671, 6925.00947243, -8007.19755945, 4388.62534545, -1032.06871257])
        else:
            if self._Benchmark in ['BM7']:
                DimDomain =  np.array([0.19,0.17,0.225])
            elif self._Benchmark in ['BM8']:
                DimDomain =  np.array([0.184,0.224,0.212])
            else:
                DimDomain =  np.array([0.07,0.07,0.12])
            self._SIM_SETTINGS = SimulationConditions(baseMaterial=Material['Water'],
                                basePPW=self._basePPW,
                                PaddingForKArray=self._PaddingForKArray,
                                bDisplay=self._bDisplay, 
                                DimDomain=DimDomain,
                                SensorSubSampling=SensorSubSampling,
                                NumberCyclesToTrackAtEnd=NumberCyclesToTrackAtEnd,
                                DispersionCorrection=[-2307.53581298, 6875.73903172, -7824.73175146, 4227.49417250, -975.22622721])
        OverwriteDuration=None
        if self._Benchmark in ['BM2','BM4','BM6']:
            #single layer of crotical
            SelM=Material['Cortical']
            self._SIM_SETTINGS.AddMaterial(SelM[0],SelM[1],SelM[2]*self._Shear,SelM[3],SelM[4]*self._Shear)
            
        elif self._Benchmark in ['BM3','BM5']:
            #composite layer of skin, cortical , trabecular and brain
            for k in ['Skin','Cortical','Trabecular','Brain']:
                SelM=Material[k]
                self._SIM_SETTINGS.AddMaterial(SelM[0],SelM[1],SelM[2]*self._Shear,SelM[3],SelM[4]*self._Shear)
        elif self._Benchmark in ['BM7','BM8']:
            #composite layer cortical and brain
            OverwriteDuration=400e-6
            for k in ['Cortical','Brain']:
                SelM=Material[k]
                self._SIM_SETTINGS.AddMaterial(SelM[0],SelM[1],SelM[2]*self._Shear,SelM[3],SelM[4]*self._Shear)
        else:
            assert(self._Benchmark=='BM1')
            #the default conditions are water only
        #########
        self._SIM_SETTINGS.UpdateConditions(AlphaCFL=self._AlphaCFL,
                                            Benchmark=self._Benchmark,
                                            OverwriteDuration=OverwriteDuration)
        ###########
            

    def Step2_PrepMaterials(self,**kwargs):
        
        print('Material properties\n',self._SIM_SETTINGS.ReturnArrayMaterial(),'[%s]' %(self._Benchmark))
        
        if self._Benchmark =='BM2':
            #single flat skull bone
            self._SIM_SETTINGS.CreateMaterialMapSingleFlatSlice(ThicknessBone=6.5e-3,LocationBone=30e-3)
        elif self._Benchmark =='BM3':
            #multi tissue type flat material
            self._SIM_SETTINGS.CreateMaterialMapCompositeFlatSlice(ThicknessMaterials=[4e-3,1.5e-3,4e-3,1e-3],Location=26e-3)
        elif self._Benchmark =='BM4':
            
            self._SIM_SETTINGS.CreateMaterialMapSingleCurvedSlice(ThicknessBone=6.5e-3,LocationBone=30e-3,SkullRadius=75e-3)
             
        elif self._Benchmark =='BM5':
            LocationBone=30e-3;
            if self._basePPW >12:
                 LocationBone+=self._SIM_SETTINGS._SpatialStep*0.75;
            self._SIM_SETTINGS.CreateMaterialMapCompositeCurvedSlice(ThicknessBone=6.5e-3,LocationBone=LocationBone,SkullRadius=75e-3,ThicknessTissues=[4e-3,1.5e-3,4e-3,1e-3])
        elif self._Benchmark in ['BM6','BM7','BM8']:
            skull_stl='../SKULL-MAPS/skull_outer.stl'
            inner_stl='../SKULL-MAPS/skull_inner.stl'
            if self._Benchmark in ['BM6','BM7']:
                affine_file='../SKULL-MAPS/affine_transform_v1.mat' 
            else:
                affine_file='../SKULL-MAPS/affine_transform_m1.mat' 
                
            if self._Benchmark in ['BM6']:
                bExtractBrain=False
            else:
                bExtractBrain=True
            
            self._SIM_SETTINGS.CreateMaterialMapFullSkull(Suffix=self._TESTNAME,
                                                         bExtractBrain=bExtractBrain,
                                                         RasterInputPath=self._RasterInputPath,
                                                         OutputPath=self._OutputPath,
                                                         TESTNAME=self._Phase+'-'+self._Benchmark+'-%iPPW' %(self._basePPW))
            

        elif self._Benchmark !='BM1':
            raise ValueError('Benchmark not supported ' + self._Benchmark)
        gc.collect()

    def Step3_PrepareSource(self):
        Suffix=os.path.join(self._OutputPath,self._TESTNAME)                     
        if self._SourceType =='SC1':
            TxRadius = 0.064 # m, 
            TxDiameter=0.064
            self._SIM_SETTINGS.CreateKArrayFUSSource(TxRadius,
                                                     TxDiameter,
                                                     Suffix=Suffix,
                                                     bSourceDisplacement=self._bSourceDisplacement)
            #a bit of sanity test
            AA=np.where(self._SIM_SETTINGS._weight_amplitudes>=1.0)
            print("index location, and location in Z = %i, %g"  % (AA[2].min(),self._SIM_SETTINGS._ZDim[AA[2].min()]))
            assert(AA[2].min()>=1.0)
            #plot source 
            self._SIM_SETTINGS.PlotWeightedAverags()
        else:
            TxDiam = 0.02 # m, circular piston
            self._SIM_SETTINGS.CreateKArrayPistonSource(TxDiam,
                                                        Suffix=Suffix,
                                                        bSourceDisplacement=self._bSourceDisplacement)
            AA=np.where(self._SIM_SETTINGS._weight_amplitudes>=1.0)
            print("index location, and location in Z = %i, %g"  % (AA[2].min(),self._SIM_SETTINGS._ZDim[AA[2].min()]))
            assert(AA[2].min()>=1.0)
            self._SIM_SETTINGS.PlotWeightedAverags()
             

    def Step4_CreateSourceSignal_and_Sensor(self):
        self._SIM_SETTINGS.CreateSource()
        if self._Benchmark in  ['BM6','BM7','BM8']:
            b3D=True
        else:
            b3D=False
        self._SIM_SETTINGS.CreateSensorMap(b3D=b3D)

    def Step5_Run_Simulation(self,COMPUTING_BACKEND=1,GPUName='SUPER',GPUNumber=0,bApplyCorrectionForDispersion=True):
        SelMapsRMSPeakList=['Pressure']
            
        self._SIM_SETTINGS.RUN_SIMULATION(COMPUTING_BACKEND=COMPUTING_BACKEND, GPUName=GPUName,GPUNumber=GPUNumber,SelMapsRMSPeakList=SelMapsRMSPeakList,
                                         bSourceDisplacement=self._bSourceDisplacement,
                                         bApplyCorrectionForDispersion=bApplyCorrectionForDispersion)
        gc.collect()

    def Step6_ExtractPhaseData(self):
        b3D= self._Benchmark in  ['BM6','BM7','BM8']
        self._SIM_SETTINGS.CalculatePhaseData(b3D=b3D)
        gc.collect()

    def Step7_PrepAndPlotData(self):
        if self._Benchmark in  ['BM6','BM7','BM8']:
            self._SIM_SETTINGS.PlotData3D()
        else:
            self._SIM_SETTINGS.PlotData()

    def Step8_ResamplingAndExport(self,bReload=False,bSkipSaveAndReturnData=False,bUseCupyToInterpolate=True):
        if self._Benchmark in  ['BM6','BM7','BM8']:
            return self._SIM_SETTINGS.ResamplingToFocusConditions3D(TESTNAME=self._TESTNAME,
                            OutputPath=self._OutputPath,
                            bReload=bReload,
                            bSkipSaveAndReturnData=bSkipSaveAndReturnData,
                            bUseCupyToInterpolate=bUseCupyToInterpolate)
        else:
            return self._SIM_SETTINGS.ResamplingToFocusConditions(TESTNAME=self._TESTNAME,
                            OutputPath=self._OutputPath,
                            bReload=bReload,
                            bSkipSaveAndReturnData=bSkipSaveAndReturnData)
        gc.collect()

    def OutPutConditions(self):
        ### Usage details

        String = 'Plese see below code implementing the complete simulation.\n'+\
                'Main highlights:\n\n'+\
                'Item  | value\n'+\
                '---- | ----\n'+\
                'PML size |  %i\n' %(self._SIM_SETTINGS._PMLThickness)+\
                'Spatial step$^*$ | $\\frac{\\lambda}{%i}$ = %3.2f mm (%i PPW)\n' %(self._SIM_SETTINGS._basePPW,\
                                                                                    self._SIM_SETTINGS._SpatialStep*1e3,\
                                                                                    self._SIM_SETTINGS._basePPW) +\
                'Final Interpolation at 0.5 mm | Linear for amplitude, nearest for phase\n'+\
                'FDTD solver | $O(2)$ temporal, $O(4)$ spatial, staggered grid\n'+\
                'Temporal step | %4.3f $\mu$s, %2.1f points-per-period\n'%(self._SIM_SETTINGS._TemporalStep*1e6,self._SIM_SETTINGS._PPP)+\
                'Adjusted CFL | %3.2f \n'%(self._SIM_SETTINGS._AdjustedCFL)+\
                'Source func. | CW-pulse for %4.1f $\mu$s\n' %(self._SIM_SETTINGS._TimeSimulation*1e6)+\
                'Amplitude method | Peak\n'+\
                'Phase method | NA in library, but it it is calculated from captured sensor data and FFT\n\n'+\
                '$^*$Spatial step chosen to produce peak pressure amplitude ~2% compared to reference simulation.'\
                
        return String

################################################################################################################
################################################################################################################
################################################################################################################
class SimulationConditions(object):
    '''
    Class implementing the low level interface to prepare the details of the simulation conditions and execute the simulation
    '''
    def __init__(self,baseMaterial=Material['Water'],
                      basePPW=9,
                      PMLThickness = 12, # grid points for perect matching layer, HIGHLY RECOMMENDED DO NOT CHANGE THIS SIZE 
                      ReflectionLimit= 1e-5, #DO NOT CHANGE THIS
                      DimDomain =  np.array([0.07,0.07,0.12]),
                      SensorSubSampling=1,
                      NumberCyclesToTrackAtEnd=2,
                      SourceAmp=60e3, # kPa
                      Frequency=500e3,
                      PaddingForKArray=0,
                      QfactorCorrection=True,
                      OffCenterBowl=4.2e-3,#distance to apply to make top edge of bowl source to sit exactly at z==0.0 mm
                      bDisplay=True,
                      DispersionCorrection=[-2307.53581298, 6875.73903172, -7824.73175146, 4227.49417250, -975.22622721]):  #coefficients to correct for values lower of CFL =1.0 in wtaer conditions.
        self._Materials=[[baseMaterial[0],baseMaterial[1],baseMaterial[2],baseMaterial[3],baseMaterial[4]]]
        self._basePPW=basePPW
        self._PMLThickness=PMLThickness
        self._ReflectionLimit=ReflectionLimit
        self._ODimDomain =DimDomain 
        self._SensorSubSampling=SensorSubSampling
        self._NumberCyclesToTrackAtEnd=NumberCyclesToTrackAtEnd
        self._TemporalStep=0.
        self._N1=0
        self._N2=0
        self._N3=0
        self._FactorConvPtoU=baseMaterial[0]*baseMaterial[1]
        self._SourceAmpPa=SourceAmp
        self._SourceAmpDisplacement=SourceAmp/self._FactorConvPtoU
        self._Frequency=Frequency
        self._weight_amplitudes=1.0
        self._PaddingForKArray=PaddingForKArray
        self._QfactorCorrection=QfactorCorrection
        self._OffCenterBowl=OffCenterBowl
        self._bDisplay=bDisplay
        self._DispersionCorrection=DispersionCorrection
        self._GMapTotal=None
        
        
    def AddMaterial(self,Density,LSoS,SSoS,LAtt,SAtt): #add material (Density (kg/m3), long. SoS 9(m/s), shear SoS (m/s), Long. Attenuation (Np/m), shear attenuation (Np/m)
        self._Materials.append([Density,LSoS,SSoS,LAtt,SAtt]);
        
        
    @property
    def Wavelength(self):
        return self._Wavelength
        
        
        
    @property
    def SpatialStep(self):
        return self._SpatialStep
        
    def UpdateConditions(self, AlphaCFL=1.0,OverwriteDuration=None,DomMaterial=0,Benchmark=''):
        '''
        Update simulation conditions
        '''
        MatArray=self.ReturnArrayMaterial()
        SmallestSOS=np.sort(MatArray[:,1:3].flatten())
        iS=np.where(SmallestSOS>0)[0]
        SmallestSOS=SmallestSOS[iS[0]]
        self._Wavelength=SmallestSOS/self._Frequency
        self._baseAlphaCFL =AlphaCFL
        print(" Wavelength, baseAlphaCFL",self._Wavelength,AlphaCFL)
        print ("smallSOS ", SmallestSOS)
        
        SpatialStep=self._Wavelength/self._basePPW
        
        dummyMaterialMap=np.zeros((10,10,MatArray.shape[0]),np.uint32)
        for n in range(MatArray.shape[0]):
            dummyMaterialMap[:,:,n]=n
        
        OTemporalStep,_,_, _, _,_,_,_,_,_=PModel.CalculateMatricesForPropagation(dummyMaterialMap,MatArray,self._Frequency,self._QfactorCorrection,SpatialStep,AlphaCFL)
        
        self.DominantMediumTemporalStep,_,_, _, _,_,_,_,_,_=PModel.CalculateMatricesForPropagation(dummyMaterialMap*0,MatArray[DomMaterial,:].reshape((1,5)),self._Frequency,self._QfactorCorrection,SpatialStep,1.0)

        TemporalStep=OTemporalStep

        print('"ideal" TemporalStep',TemporalStep)
        print('"ideal" DominantMediumTemporalStep',self.DominantMediumTemporalStep)

        #now we make it to be an integer division of the period
        TemporalStep=1/self._Frequency/(np.ceil(1/self._Frequency/TemporalStep)) # we make it an integer of the period
        #and back to SpatialStep
        print('"int fraction" TemporalStep',TemporalStep)
        print('"CFL fraction relative to dominant only conditions',TemporalStep/self.DominantMediumTemporalStep)
        
        self._PPP=int(np.round(1/self._Frequency/TemporalStep))
        self._AdjustedCFL=TemporalStep/OTemporalStep*AlphaCFL
        print("adjusted AlphaCL, PPP",self._AdjustedCFL,self._PPP)
        
        self._SpatialStep=SpatialStep
        self._TemporalStep=TemporalStep
        
        self._N1=int(np.ceil(self._ODimDomain[0]/self._SpatialStep)+2*self._PMLThickness)
        if self._N1%2==0:
            self._N1+=1
        self._N2=int(np.ceil(self._ODimDomain[1]/self._SpatialStep)+2*self._PMLThickness)
        if self._N2%2==0:
            self._N2+=1
        self._N3=int(np.ceil(self._ODimDomain[2]/self._SpatialStep)+2*self._PMLThickness)
        #this helps to avoid the "bleeding" from the low values of the PML in the exported files 
        self._N3+=self._PaddingForKArray+int(np.floor(0.5e-3/SpatialStep)) 
        
        print('Domain size',self._N1,self._N2,self._N3)
        self._DimDomain=np.zeros((3))
        self._DimDomain[0]=self._N1*SpatialStep
        self._DimDomain[1]=self._N2*SpatialStep
        self._DimDomain[2]=self._N3*SpatialStep
        if OverwriteDuration is not None:
            self._TimeSimulation=OverwriteDuration
        else:
            self._TimeSimulation= np.sqrt(self._DimDomain[0]**2+\
                                          self._DimDomain[1]**2+self._DimDomain[2]**2)/MatArray[0,1] #time to cross one corner to another
            self._TimeSimulation=np.floor(self._TimeSimulation/self._TemporalStep)*self._TemporalStep
            
        TimeVector=np.arange(0.0,self._TimeSimulation,self._TemporalStep)
        ntSteps=(int(TimeVector.shape[0]/self._PPP)+1)*self._PPP
        self._TimeSimulation=self._TemporalStep*ntSteps
        TimeVector=np.arange(0.0,self._TimeSimulation,self._TemporalStep)
        assert(TimeVector.shape[0]==ntSteps)
        print('PPP for sensors ', self._PPP/self._SensorSubSampling)
        assert((self._PPP%self._SensorSubSampling)==0)
        
  
        nStepsBack=int(self._NumberCyclesToTrackAtEnd*self._PPP)
        self._SensorStart=int((TimeVector.shape[0]-nStepsBack)/self._SensorSubSampling)
        
        self._MaterialMap=np.zeros((self._N1,self._N2,self._N3),np.uint32) # note the 32 bit size

        
        print('PPP, Duration simulation',np.round(1/self._Frequency/TemporalStep),self._TimeSimulation*1e6)
        
        print('Number of steps sensor',np.floor(self._TimeSimulation/self._TemporalStep/self._SensorSubSampling)-self._SensorStart)
        
        self._XDim=(np.arange(self._N1)*self._SpatialStep)*1e3 #mm
        self._XDim-=self._XDim.mean()
        self._YDim=(np.arange(self._N2)*self._SpatialStep)*1e3 #mm
        self._YDim-=self._YDim.mean()
        self._ZDim=(np.arange(self._N3)*SpatialStep-(self._PMLThickness)*self._SpatialStep)*1e3 - self._PaddingForKArray*SpatialStep*1e3#mm
        
        ## We also initialize the target 6 PPW conditions for the comparison
        assert(Benchmark in ['BM1','BM2','BM3','BM4','BM5','BM6','BM7','BM8'])
        if Benchmark in  ['BM1','BM2','BM3','BM4','BM5']:
            self._FZDim=np.arange(0,0.1205,0.0005)*1e3
            self._FXDim=np.round(np.arange(-0.035,0.0355,0.0005),5)*1e3
        elif Benchmark=='BM6':
            self._FZDim=np.arange(0,0.1205,0.0005)*1e3
            self._FXDim=np.round(np.arange(-0.035,0.0355,0.0005),5)*1e3
            self._FYDim=np.round(np.arange(-0.035,0.0355,0.0005),5)*1e3
        elif Benchmark=='BM7':
            self._FZDim=np.round(np.arange(0,0.2255,0.0005),5)*1e3
            self._FXDim=np.round(np.arange(-0.095,0.0955,0.0005),5)*1e3
            self._FYDim=np.round(np.arange(-0.085,0.0855,0.0005),5)*1e3
        else:
            self._FZDim=np.round(np.arange(0,0.2125,0.0005),5)*1e3
            self._FXDim=np.round(np.arange(-0.092,0.0925,0.0005),5)*1e3
            self._FYDim=np.round(np.arange(-0.112,0.1125,0.0005),5)*1e3
            
        
        
    def ReturnArrayMaterial(self):
        return np.array(self._Materials)
        
    def CreateMaterialMapSingleFlatSlice(self,ThicknessBone=6.5e-3,LocationBone=30e-3):
        ThicknessBoneSteps=int(np.round(ThicknessBone/self._SpatialStep))
        LocationBoneSteps=int(np.round(LocationBone/self._SpatialStep))+self._PMLThickness+self._PaddingForKArray
        print("LocationBoneSteps",LocationBoneSteps,self._ZDim[LocationBoneSteps])
        self._MaterialMap[:,:,LocationBoneSteps:LocationBoneSteps+ThicknessBoneSteps]=1 #Material 1 is Cortical Bone
        if self._bDisplay:
            plt.figure(figsize=(12,8))
            plt.subplot(1,2,1)
            plt.imshow(self._MaterialMap[int(self._N1/2),:,:].T,cmap=plt.cm.gray,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);

    def CreateMaterialMapCompositeFlatSlice(self,ThicknessMaterials=[4e-3,1.5e-3,4e-3,1e-3],Location=26e-3): 
        MatType=1
        LocationSteps=int(np.round(Location/self._SpatialStep))+self._PMLThickness+self._PaddingForKArray
        for n,t in enumerate(ThicknessMaterials):
            ThicknessSteps=int(np.round(t/self._SpatialStep))
            if n==3:
                self._MaterialMap[:,:,LocationSteps:LocationSteps+ThicknessSteps]=2 
            else:
                self._MaterialMap[:,:,LocationSteps:LocationSteps+ThicknessSteps]=MatType 
                MatType+=1
            LocationSteps+=ThicknessSteps
        self._MaterialMap[:,:,LocationSteps:]=MatType

        if self._bDisplay:
            plt.figure(figsize=(14,6))
            plt.subplot(1,2,1)
            plt.imshow(self._MaterialMap[int(self._N1/2),:,:].T,cmap=plt.cm.Set3,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);
            plt.colorbar()

            plt.subplot(1,2,2)
            plt.imshow(self._MaterialMap[int(self._N1/2),:,:].T,cmap=plt.cm.Set3,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);
            plt.colorbar()
            plt.ylim(39,24)
            plt.xlim(-5,5)
            plt.suptitle('Material Map')
    
    
    
    def CreateMaterialMapSingleCurvedSlice(self,ThicknessBone=6.5e-3,LocationBone=30e-3,SkullRadius=75e-3):
        #Create single curved skull type interface
        LocationBoneSteps=int(np.round(LocationBone/self._SpatialStep))+self._PMLThickness+self._PaddingForKArray
        
        CenterSkull=[0,0,(SkullRadius+LocationBone-self._SpatialStep*0.75)*1e3] 
        
        Mask=self.MakeSphericalMask(Radius=SkullRadius*1e3,Center=CenterSkull)
        AA=np.where(Mask)
        print(AA[2].min(), LocationBoneSteps)
        assert(AA[2].min()== LocationBoneSteps)
        BelowMask=self.MakeSphericalMask(Radius=(SkullRadius-ThicknessBone)*1e3,Center=CenterSkull)
        Mask=np.logical_xor(Mask,BelowMask)

        self._MaterialMap[Mask]=1 #Material 1 is Cortical Bone
        if self._bDisplay:
            print('Display curved')
            plt.figure(figsize=(12,8))
            plt.subplot(1,2,1)
            plt.imshow(self._MaterialMap[int(self._N1/2),:,:].T,cmap=plt.cm.gray,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);
            plt.subplot(1,2,2)
            plt.imshow(self._MaterialMap[:,int(self._N2/2),:].T,cmap=plt.cm.gray,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);

    def CreateMaterialMapCompositeCurvedSlice(self,ThicknessBone=6.5e-3,LocationBone=30e-3,SkullRadius=75e-3,ThicknessTissues=[4e-3,1.5e-3,4e-3,1e-3],adjustCenter=0.75):
        
        LocationBoneSteps=int(np.round(LocationBone/self._SpatialStep))+self._PMLThickness+self._PaddingForKArray
        
        CenterSkull=[0,0,(SkullRadius+LocationBone-self._SpatialStep*adjustCenter)*1e3] 

        OuterTableROC=SkullRadius

        TissueRadius=np.array([OuterTableROC+ThicknessTissues[0], #skin
                    OuterTableROC, #outer table
                    OuterTableROC-ThicknessTissues[1], #diploe
                    OuterTableROC-ThicknessTissues[1]-ThicknessTissues[2], # inner table
                    OuterTableROC-ThicknessTissues[1]-ThicknessTissues[2]-ThicknessTissues[3]]) #brain

        print('TissueRadius before rounding in steps',TissueRadius)
        #we round each interface in terms of steps
        TissueRadius=np.round(TissueRadius/self._SpatialStep)*self._SpatialStep
        print('TissueRadius after rounding in steps',TissueRadius)

        for n in range(len(TissueRadius)-1):
            Mask=self.MakeSphericalMask(Radius=TissueRadius[n]*1e3,Center=CenterSkull)
            if n==1: #sanitycheck for the bonelocation
                AA=np.where(Mask)
                print(AA[2].min(), LocationBoneSteps)
                assert(AA[2].min()== LocationBoneSteps)
           
            BelowMask=self.MakeSphericalMask(Radius=TissueRadius[n+1]*1e3,Center=CenterSkull)

            Mask=np.logical_xor(Mask,BelowMask)
            if n==3: #the inner table
                self._MaterialMap[Mask]=2 
            else:
                self._MaterialMap[Mask]=n+1 
                
        #And we use the last below mask to make it the brain tissue
        self._MaterialMap[BelowMask]=4
        if self._bDisplay:
            plt.figure(figsize=(12,8))
            plt.subplot(1,2,1)
            plt.imshow(self._MaterialMap[int(self._N1/2),:,:].T,cmap=plt.cm.Set3,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(self._MaterialMap[:,int(self._N2/2),:].T,cmap=plt.cm.Set3,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);
            plt.colorbar()

            plt.ylim(42,25)
            plt.xlim(-10,10)
################################
    def CreateMaterialMapFullSkull(self,Suffix='',
                                      bExtractBrain=False,
                                      bClearOrphansBone=False,
                                      RasterInputPath='',
                                      OutputPath='',
                                      TESTNAME='test'):
        SpatialStep=self._SpatialStep*1e3 #we use in mm
        affine_transform=loadmat(affine_file)['affine_transform']
        
        x_vec=self._XDim
        y_vec=self._YDim
        z_vec=self._ZDim
        
        preSavedName=os.path.join(OutputPath,TESTNAME+'-SkullMask.npz')
        
        bReloadVoxelization=True
        
        if os.path.isfile(preSavedName):
            PrevData=np.load(preSavedName)
            PrevMask=PrevData['Mask']
            if np.all(np.array(PrevMask.shape)==np.array(self._MaterialMap.shape)):
                bReloadVoxelization=False
                print('Reloading from ',preSavedName)
                self._MaterialMap[:,:,:]=PrevMask
                AllHeadMask=PrevData['AllHeadMask']
                tOUTName=TESTNAME+'-SkullOuter-GMap.npz'
                if not os.path.isfile(tOUTName) and bCalculateForSPP: #at least at this point, there is an GMAp file, but if there isn't we have to recalculate a few objets
                    bReloadVoxelization=True
        
        if bReloadVoxelization: #we just load the precalculated mask to avoid sources of discrepancies
            if 'BM6' in TESTNAME:
                infix='bm6'
            elif 'BM7' in TESTNAME:
                infix='bm7'
            else:
                infix='bm8'
            if '6PPW' in TESTNAME:
                resfix='0.5'
            elif '9PPW' in TESTNAME:
                resfix='0.33333'
            elif '12PPW' in TESTNAME:
                resfix='0.25'
            elif '15PPW' in TESTNAME:
                resfix='0.2'
            elif '18PPW' in TESTNAME:
                resfix='0.16667'
            else:
                resfix='0.125' #24 PPW
                
            vfname=os.path.join(RasterInputPath,'skull_mask_'+infix+'_dx_'+resfix+'mm.mat')
            print('Loadinf surf2vol data ',vfname)
            vol2surfdata=mat73.loadmat(vfname)
            MLskull_mask=vol2surfdata['skull_mask'].T
            MLbrain_mask=vol2surfdata['brain_mask'].T
            MLzi=vol2surfdata['xi']
            MLxi=vol2surfdata['zi']
            MLyi=vol2surfdata['yi']
            RatioPPW=np.round((self._XDim[1]-self._XDim[0])/(MLxi[1]-MLxi[0]),5)
            if RatioPPW==1.0: #we have the exact resolution
                print('we have the same resolution')
                MLskull_mask_ds=np.zeros(self._MaterialMap.shape,np.bool)
                MLbrain_mask_ds=np.zeros(self._MaterialMap.shape,np.bool)
                
                OrigX=np.argmin(np.abs(self._XDim-MLxi[0]))
                OrigY=np.argmin(np.abs(self._YDim-MLyi[0]))
                OrigZ=np.argmin(np.abs(self._ZDim-MLzi[0]))
                MLskull_mask_ds[OrigX:OrigX+MLskull_mask.shape[0],
                                OrigY:OrigY+MLskull_mask.shape[1],
                                OrigZ:OrigZ+MLskull_mask.shape[2]]=MLskull_mask
                MLbrain_mask_ds[OrigX:OrigX+MLskull_mask.shape[0],
                                OrigY:OrigY+MLskull_mask.shape[1],
                                OrigZ:OrigZ+MLskull_mask.shape[2]]=MLbrain_mask
                
                
            else:
                print('we have diff resolution')
                #this can happen as we use trabecular as the lowest wavelenght, so we need to interpolate
        
            
                print('RatioPPW',RatioPPW)
                OrigX=np.argmin(np.abs(MLxi-self._XDim[0]))
                OrigX+=(self._XDim[0]-MLxi[OrigX])/(MLxi[1]-MLxi[0])
                OrigY=np.argmin(np.abs(MLyi-self._YDim[0]))
                OrigY+=(self._YDim[0]-MLxi[OrigY])/(MLyi[1]-MLyi[0])
                OrigZ=np.argmin(np.abs(MLzi-self._ZDim[0]))
                OrigZ+=(self._ZDim[0]-MLzi[OrigZ])/(MLzi[1]-MLzi[0])
            
                gFXDim=(cupy.arange(self._XDim.shape[0])*RatioPPW).astype(np.float32)+OrigX
                gFYDim=(cupy.arange(self._YDim.shape[0])*RatioPPW).astype(np.float32)+OrigY
                gFZDim=(cupy.arange(self._ZDim.shape[0])*RatioPPW).astype(np.float32)+OrigZ
                FRefY,FRefX,FRefZ=cupy.meshgrid(gFYDim,gFXDim,gFZDim)
                gMLskull_mask=cupy.asarray(MLskull_mask)
                gMLskull_mask_ds=cndimage.map_coordinates(gMLskull_mask, cupy.array([FRefX.ravel(),FRefY.ravel(),FRefZ.ravel()]),order=0)
                gMLskull_mask_ds=gMLskull_mask_ds.reshape(FRefX.shape)
                MLskull_mask_ds=gMLskull_mask_ds.get()
                self._MaterialMap[MLskull_mask_ds]=1
                
                gMLbrain_mask=cupy.asarray(MLbrain_mask)
                gMLbrain_mask_ds=cndimage.map_coordinates(gMLbrain_mask, cupy.array([FRefX.ravel(),FRefY.ravel(),FRefZ.ravel()]),order=0)
                gMLbrain_mask_ds=gMLbrain_mask_ds.reshape(FRefX.shape)
                MLbrain_mask_ds=gMLbrain_mask_ds.get()
                    
            AllHeadMask=np.logical_or(MLbrain_mask_ds,MLskull_mask_ds)
            self._MaterialMap[MLskull_mask_ds]=1
            if bExtractBrain:
                self._MaterialMap[MLbrain_mask_ds]=2
            np.savez_compressed(preSavedName,Mask=self._MaterialMap,AllHeadMask=AllHeadMask,
                                   XDim=self._XDim,YDim=self._YDim,ZDim=self._ZDim)
    

        if self._bDisplay:
            ## use this approach to ensure the notebook will save the rendering inside the notebook
            plt.figure(figsize=(16,8))
            plt.subplot(1,2,1)
            img = mpimg.imread(TESTNAME+'-Rendering-1.png')
            imgplot = plt.imshow(img)
            plt.subplot(1,2,2)
            img = mpimg.imread(TESTNAME+'-Rendering-2.png')
            imgplot = plt.imshow(img)
            
            plt.figure(figsize=(16,5))
            plt.subplot(1,2,1)
            plt.imshow(self._MaterialMap[int(self._N1/2),:,:].T,cmap=plt.cm.gray,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.gca().set_aspect(1.0)
            plt.ylim(40,20)          
            plt.xlim(-10,10)
            
            plt.subplot(1,2,2)
            plt.imshow(self._MaterialMap[:,int(self._N2/2),:].T,cmap=plt.cm.gray,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.gca().set_aspect(1.0)
            plt.ylim(40,20)          
            plt.xlim(-10,10)
            
    
    def CalculatekArrayWeigthedArraysBowl(self,TxRadius,
                                          TxDiameter,
                                          BLITolerance=0.05,
                                          UpsamplingRate=10,
                                          Suffix=''):
        global MatlabEng
        '''
        Invoke k-Wave functions to calculate weigthed array for bowl sources
        '''
        PrepDatakWaveArray={}
        PrepDatakWaveArray['N1']=self._N1
        PrepDatakWaveArray['N2']=self._N2
        PrepDatakWaveArray['N3']=self._N3
        PrepDatakWaveArray['SpatialStep']=self._SpatialStep
        PrepDatakWaveArray['TxRadius']=TxRadius
        PrepDatakWaveArray['TxDiameter']=TxDiameter
        PrepDatakWaveArray['SourceLocatonGridPoints']=np.array([int(self._N1/2)+1,int(self._N2/2)+1,self._PMLThickness+1+self._PaddingForKArray]) #in 1-index convention
        PrepDatakWaveArray['BLITolerance']=BLITolerance
        PrepDatakWaveArray['UpsamplingRate']=UpsamplingRate
        PrepDatakWaveArray['bDisplay']=0.0

        bRecalculate=False
        
        MiddleFile=Suffix+'-kArray.h5'
        InFile=Suffix+'-kArrayIn.h5'
               
        if os.path.isfile(MiddleFile):
            SavedData=ReadFromH5py(MiddleFile)
            for k in PrepDatakWaveArray:
                if np.any(SavedData[k]!=PrepDatakWaveArray[k]):
                    bRecalculate=True
                    break
        else:
            bRecalculate=True

        if bRecalculate:
            if MatlabEng is None:
                import matlab.engine
                MatlabEng = matlab.engine.start_matlab()
            SaveToH5py(PrepDatakWaveArray,InFile)
            print('Calling Matlab function to calculate kArray weighted array')
            #if dimensions of grid, source and location hasn't changed 
            MatlabEng.CalculateKGridArrayBowl(InFile,nargout=0)
            outfile=Suffix+'-kArrayOut.mat'
            weight_amplitudes=mat73.loadmat(outfile)['source_weights']
            PrepDatakWaveArray['weight_amplitudes']=weight_amplitudes
            SaveToH5py(PrepDatakWaveArray,MiddleFile)
        else:
            print('loading pre calculated weighted array')
            weight_amplitudes=SavedData['weight_amplitudes']

        self._weight_amplitudes=weight_amplitudes
        
    def CalculatekArrayWeigthedArraysPiston(self,TxDiameter,BLITolerance=0.05,UpsamplingRate=10,Suffix=''):
        '''
        Invoke k-Wave functions to calculate weigthed array for bowl sources
        '''
        global MatlabEng
        PrepDatakWaveArray={}
        PrepDatakWaveArray['N1']=self._N1
        PrepDatakWaveArray['N2']=self._N2
        PrepDatakWaveArray['N3']=self._N3
        PrepDatakWaveArray['SpatialStep']=self._SpatialStep
        PrepDatakWaveArray['TxDiameter']=TxDiameter
        PrepDatakWaveArray['SourceLocatonGridPoints']=np.array([int(self._N1/2)+1,int(self._N2/2)+1,self._PMLThickness+1+self._PaddingForKArray]) #in 1-index convention
        PrepDatakWaveArray['BLITolerance']=BLITolerance
        PrepDatakWaveArray['UpsamplingRate']=UpsamplingRate
        PrepDatakWaveArray['bDisplay']=0.0

       
        bRecalculate=False
        
        MiddleFile=Suffix+'-kArray.h5'
        InFile=Suffix+'-kArrayIn.h5'
               
        if os.path.isfile(MiddleFile):
            SavedData=ReadFromH5py(MiddleFile)
            for k in PrepDatakWaveArray:
                if np.any(SavedData[k]!=PrepDatakWaveArray[k]):
                    bRecalculate=True
                    break
        else:
            SaveToH5py(PrepDatakWaveArray,InFile)
            bRecalculate=True

        if bRecalculate:
            if MatlabEng is None:
                import matlab.engine
                MatlabEng = matlab.engine.start_matlab()
            print('Calling Matlab function to calculate kArray weighted array')
            #if dimensions of grid, source and location hasn't changed 
            MatlabEng.CalculateKGridArrayDisc(InFile,nargout=0)
            outfile=Suffix+'-kArrayOut.mat'
            weight_amplitudes=mat73.loadmat(outfile)['source_weights']
            PrepDatakWaveArray['weight_amplitudes']=weight_amplitudes
            SaveToH5py(PrepDatakWaveArray,MiddleFile)
        else:
            print('loading pre calculated weighted array')
            weight_amplitudes=SavedData['weight_amplitudes']

        self._weight_amplitudes=weight_amplitudes
    
    def MakeCircularSource(self,Diameter):
        #simple defintion of a circular source centred in the domain
        XDim=np.arange(self._N1)*self._SpatialStep
        YDim=np.arange(self._N2)*self._SpatialStep
        XDim-=XDim.mean()
        YDim-=YDim.mean()
        XX,YY=np.meshgrid(XDim,YDim)
        MaskSource=(XX**2+YY**2)<=(Diameter/2.0)**2
        return (MaskSource*1.0).astype(np.uint32)
        
    def CreatePistonSource(self,TxDiam):
        SourceMask=self.MakeCircularSource(TxDiam)
        if self._bDisplay:
            plt.imshow(SourceMask,cmap=plt.cm.gray);
            plt.title('Circular source map')
        
        self._SourceMap=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        LocZ=self._PMLThickness
        self._SourceMap[:,:,LocZ]=SourceMask 

        self._Ox=np.zeros((self._N1,self._N2,self._N3))
        self._Oy=np.zeros((self._N1,self._N2,self._N3))
        self._Oz=np.zeros((self._N1,self._N2,self._N3))
        self._Oz[self._SourceMap>0]=1 #only Z has a value of 1
        
        self._weight_amplitudes=np.zeros(self._SourceMap.shape)
        self._weight_amplitudes[self._SourceMap>0]=1.0

        if self._bDisplay:
            plt.figure(figsize=(12,8))
            plt.subplot(1,2,1)
            plt.imshow(self._SourceMap[int(self._N1/2),:,:].T,cmap=plt.cm.gray,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()]);

            plt.subplot(1,2,2)
            plt.imshow(self._SourceMap[:,int(self._N2/2),:].T,cmap=plt.cm.gray,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()]);



    def MakeFocusingSource(self,TxRadius,TxDiameter,Angle=0,LocZ=0,Center=[0,0,0]):
        #simple defintion of a focusing source centred in the domain, 
        #please note this is not a bullet-proof solution as it may not work for all cases
        AdjustmentZ=0.0
        while(True):
            XDim=np.arange(self._N1)*self._SpatialStep
            YDim=np.arange(self._N2)*self._SpatialStep
            ZDim=np.arange(self._N3)*self._SpatialStep
            XDim-=XDim.mean()+Center[0]
            YDim-=YDim.mean()+Center[1]
            ZDim-=ZDim.mean()+Center[2]+AdjustmentZ+LocZ*self._SpatialStep
            
            XX,YY,ZZ=np.meshgrid(YDim,XDim,ZDim)#note we have to invert this because how meshgrid works
            Depth=np.sqrt(TxRadius**2-(TxDiameter/2.0)**2)
            cX=np.argmin(np.abs(XDim))
            cZ=np.argmin(np.abs(ZDim))
            
            MaskSource=np.zeros((self._N1,self._N2,self._N3),np.bool)
            FillMask=np.zeros((self._N1,self._N2,self._N3))
            
            for n,y in enumerate(YDim):
                if np.abs(y)<=TxRadius:
                    cDiam=int(np.ceil((TxRadius)*np.sin(np.arccos(y/TxRadius))/self._SpatialStep))
                    rr, cc = circle_perimeter(cX,cZ,cDiam,shape=(self._N1,self._N3))
                    MaskSource[rr,n,cc]=np.True_
                    rr,cc=disk((cX,cZ),cDiam+1,shape=(self._N1,self._N3))
                    FillMask[rr,n,cc]=1
                    
            
            FillMask[ZZ<=-Depth]=0.
            #instead of rotating the arc, we rotate the mask that will negate the perimeter to be turned off
            if Angle!=0.:
                for n in range(N2):
                    FillMask[:,n,:]=rotate(FillMask[:,n,:],Angle,preserve_range=True)
                
            MaskSource[FillMask!=0]=False
            AA=np.where(MaskSource>0)
            if(AA[2].min()==LocZ):
                break
            if AA[2].min()>LocZ:
                AdjustmentZ-=self._SpatialStep/2
            else:
                AdjustmentZ+=self._SpatialStep/2
        
        YY+=self._SpatialStep/2
        ZZ+=self._SpatialStep/2
                
        #since the sphere mask is 0-centred, the orientation vectors in each point is straighforward
        OxOyOz=np.vstack((-XX.flatten(),-YY.flatten(),-ZZ.flatten())).T
        #and we just normalize
        OxOyOz/=np.tile( np.linalg.norm(OxOyOz,axis=1).reshape(OxOyOz.shape[0],1),[1,3])
        Ox=OxOyOz[:,1].reshape(XX.shape) 
        Oy=OxOyOz[:,0].reshape(XX.shape)
        Oz=OxOyOz[:,2].reshape(XX.shape)    
        Ox[MaskSource==False]=0
        Oy[MaskSource==False]=0
        Oz[MaskSource==False]=0
        
       
        return MaskSource.astype(np.uint32),Ox,Oy,Oz,[Center[0],Center[1],Center[2]+AdjustmentZ]

    def GetDisplacementFromFocusingMask(self,MaskSource,Center,LocZ=0):
        #simple defintion of a focusing source centred in the domain, 
        #please note this is not a bullet-proof solution as it may not work for all cases
        XDim=np.arange(self._N1)*self._SpatialStep
        YDim=np.arange(self._N2)*self._SpatialStep
        ZDim=np.arange(self._N3)*self._SpatialStep
        XDim-=XDim.mean()+Center[0]
        YDim-=YDim.mean()+Center[1]
        ZDim-=ZDim.mean()+Center[2]+LocZ*self._SpatialStep
            
        XX,YY,ZZ=np.meshgrid(YDim,XDim,ZDim)#note we have to invert this because how meshgrid works
        XX+=self._SpatialStep/2
        YY+=self._SpatialStep/2
        ZZ+=self._SpatialStep/2
                
        #since the sphere mask is 0-centred, the orientation vectors in each point is straighforward
        OxOyOz=np.vstack((-XX.flatten(),-YY.flatten(),-ZZ.flatten())).T
        #and we just normalize
        OxOyOz/=np.tile( np.linalg.norm(OxOyOz,axis=1).reshape(OxOyOz.shape[0],1),[1,3])
        Ox=OxOyOz[:,1].reshape(XX.shape) 
        Oy=OxOyOz[:,0].reshape(XX.shape)
        Oz=OxOyOz[:,2].reshape(XX.shape)    
        Ox[self._SourceMap==0]=0
        Oy[self._SourceMap==0]=0
        Oz[self._SourceMap==0]=0
        
       
        return Ox,Oy,Oz
        
        
    def CreateShellFUSSource(self,TxRadius,TxDiameter):
        self._SourceMap,\
        self._Ox,\
        self._Oy,\
        self._Oz,_=self.MakeFocusingSource(TxRadius,TxDiameter,Center=[0,0,self._OffCenterBowl],LocZ=self._PMLThickness+self._PaddingForKArray)
        self._weight_amplitudes=np.zeros(self._SourceMap.shape)
        self._weight_amplitudes[self._SourceMap>0]=1.0
        
        
    def CreateKArrayFUSSource(self, TxRadius, TxDiameter, 
                              BLITolerance=0.05, 
                              UpsamplingRate=10, 
                              Suffix='',
                              bSourceDisplacement=False):
    
        self.CalculatekArrayWeigthedArraysBowl(TxRadius, TxDiameter, 
                                               BLITolerance=BLITolerance,
                                               UpsamplingRate=UpsamplingRate,
                                               Suffix=Suffix)
        
        LocZBackTx=np.where(self._weight_amplitudes>1.0)[2].min()

        self._SourceMap=(self._weight_amplitudes!=0).astype(np.uint32)
        self._SourceMap[:,:,:self._PMLThickness]=0
        if bSourceDisplacement:
            #we use only the shell source calculation to calculate the center
            _,_,_,_,NewCenter=self.MakeFocusingSource(TxRadius,TxDiameter,LocZ=LocZBackTx)
            self._Ox,self._Oy,self._Oz=self.GetDisplacementFromFocusingMask(self._SourceMap,NewCenter,LocZ=LocZBackTx)


    def CreateKArrayPistonSource(self,TxDiameter,
                                 BLITolerance=0.05,
                                 UpsamplingRate=10,
                                 Suffix='',
                                 bSourceDisplacement=False):
    
        self.CalculatekArrayWeigthedArraysPiston(TxDiameter,BLITolerance=BLITolerance,UpsamplingRate=UpsamplingRate,Suffix=Suffix)

        self._SourceMap=(self._weight_amplitudes!=0).astype(np.uint32)
        self._SourceMap[:,:,:self._PMLThickness]=0
        
        if bSourceDisplacement:
            self._Ox=np.zeros(self._SourceMap.shape)
            self._Ox[self._SourceMap>0]=1.0
            self._Oy=np.zeros(self._SourceMap.shape)
            self._Oz=np.zeros(self._SourceMap.shape)
        
        
    def CreateSource(self,ramp_length=4):
        LengthSource=np.floor(self._TimeSimulation/(1.0/self._Frequency))*1/self._Frequency
        TimeVectorSource=np.arange(0,LengthSource+self._TemporalStep,self._TemporalStep)
        #we do as in k-wave to create a ramped signal
        
        ramp_length_points = int(np.round(ramp_length/self._Frequency/self._TemporalStep))
        ramp_axis =np.arange(0,np.pi,np.pi/ramp_length_points)

        # create ramp using a shifted cosine
        ramp = (-np.cos(ramp_axis) + 1) * 0.5
        ramp_length_points=len(ramp)

        PulseSource = np.sin(2*np.pi*self._Frequency*TimeVectorSource)
        PulseSource[:int(ramp_length_points)]*=ramp
        if self._bDisplay:
            plt.figure()
            plt.plot(TimeVectorSource*1e6,PulseSource)
            plt.title('CW signal')

        #note we need expressively to arrange the data in a 2D array
        self._PulseSource=np.reshape(PulseSource,(1,len(TimeVectorSource))) 
        
    def CreateSensorMap(self,b3D=False):
        SensorMap=np.zeros((self._N1,self._N2,self._N3),np.uint32)
        if b3D:
            SensorMap[self._PMLThickness:-self._PMLThickness,self._PMLThickness:-self._PMLThickness,self._PMLThickness:-self._PMLThickness]=1
        else:
            SensorMap[self._PMLThickness:-self._PMLThickness,int(self._N2/2),self._PMLThickness:-self._PMLThickness]=1
        self._SensorMap=SensorMap
        print('Number of sensors ', self._SensorMap.sum())
        if self._bDisplay:
            plt.figure()
            plt.imshow(SensorMap[:,int(self._N2/2),:].T,cmap=plt.cm.gray)
            plt.title('Sensor map location')
        
        
    def PlotWeightedAverags(self):
        if self._bDisplay:
            plt.figure(figsize=(12,12))
            plt.subplot(1,2,1)
            plt.imshow(self._weight_amplitudes[int(self._N1/2),:,:].T,cmap=plt.cm.jet,extent=[self._YDim.min(),self._YDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.colorbar()
            plt.xlim(-10,10)
            plt.ylim(10,-10)

            plt.subplot(1,2,2)
            plt.imshow(self._weight_amplitudes[:,int(self._N2/2),:].T,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()]);
        
            
        
    def RUN_SIMULATION(self,COMPUTING_BACKEND=1,GPUName='SUPER',GPUNumber=0,SelMapsRMSPeakList=['Vx','Vy','Vz','Pressure'],bSourceDisplacement=True,bApplyCorrectionForDispersion=True):
        MaterialList=self.ReturnArrayMaterial()

        if bSourceDisplacement:
            TypeSource=0 
            Ox=self._Ox*self._weight_amplitudes*np.cos(np.pi*2*self._Frequency*self._TemporalStep/2)
            Oy=self._Oy*self._weight_amplitudes*np.cos(np.pi*2*self._Frequency*self._TemporalStep/2)
            Oz=self._Oz*self._weight_amplitudes*np.cos(np.pi*2*self._Frequency*self._TemporalStep/2)
            PulseSource=self._PulseSource*self._SourceAmpDisplacement
        else:
            TypeSource=2 #stress source
            Ox=self._weight_amplitudes*np.cos(np.pi*2*self._Frequency*self._TemporalStep/2)
            Oy=np.array([1])
            Oz=np.array([1])
            PulseSource=-self._PulseSource*self._SourceAmpPa

        
        self._Sensor,self._LastMap,self._DictPeakValue,self._InputParam=PModel.StaggeredFDTD_3D_with_relaxation(
                                                             self._MaterialMap,
                                                             MaterialList,
                                                             self._Frequency,
                                                             self._SourceMap,
                                                             PulseSource,
                                                             self._SpatialStep,
                                                             self._TimeSimulation,
                                                             self._SensorMap,
                                                             Ox=Ox,
                                                             Oy=Oy,
                                                             Oz=Oz,
                                                             NDelta=self._PMLThickness,
                                                             DT=self._TemporalStep,
                                                             ReflectionLimit=self._ReflectionLimit,
                                                             COMPUTING_BACKEND=COMPUTING_BACKEND,
                                                             USE_SINGLE=True,
                                                             SelMapsRMSPeakList=SelMapsRMSPeakList,
                                                             SelMapsSensorsList=['Pressure'],
                                                             SelRMSorPeak=2,
                                                             DefaultGPUDeviceName=GPUName,
                                                             DefaultGPUDeviceNumber=GPUNumber,
                                                             AlphaCFL=1.0,
                                                             TypeSource=TypeSource,
                                                             QfactorCorrection=self._QfactorCorrection,
                                                             SensorSubSampling=self._SensorSubSampling,
                                                             SensorStart=self._SensorStart)
        
                
        if bApplyCorrectionForDispersion:
            CFLWater=self._TemporalStep/self.DominantMediumTemporalStep
            ExpectedError=np.polyval(self._DispersionCorrection,CFLWater)
            Correction=100.0/(100.0-ExpectedError)
            print('CFLWater only, ExpectedError, Correction', CFLWater,ExpectedError,Correction)
            for k in self._LastMap:
                self._LastMap[k]*=Correction
            for k in self._DictPeakValue:
                self._DictPeakValue[k]*=Correction
            for k in self._Sensor:
                if k=='time':
                    continue
                self._Sensor[k]*=Correction

    
    def CalculatePhaseData(self,b3D=False):
        if b3D:
            t0=time.time()
            self._PhaseMap=np.zeros((self._N1,self._N2,self._N3))
            self._PressMapFourier=np.zeros((self._N1,self._N2,self._N3))
            self._PressMapPeak=np.zeros((self._N1,self._N2,self._N3))
        else:
            self._PhaseMap=np.zeros((self._N1,self._N3))
            self._PressMapFourier=np.zeros((self._N1,self._N3))
            self._PressMapPeak=np.zeros((self._N1,self._N3))
        time_step = np.diff(self._Sensor['time']).mean() #remember the sensor time vector can be different from the input source
        
        assert((self._Sensor['time'].shape[0]%(self._PPP/self._SensorSubSampling))==0)

        freqs = np.fft.fftfreq(self._Sensor['time'].size, time_step)
        IndSpectrum=np.argmin(np.abs(freqs-self._Frequency)) # frequency entry closest to 500 kHz
        if np.isfortran(self._Sensor['Pressure']):
            self._Sensor['Pressure']=np.ascontiguousarray(self._Sensor['Pressure'])
        index=self._InputParam['IndexSensorMap']
        
        if b3D:
            nStep=100000
            for n in range(0,self._Sensor['Pressure'].shape[0],nStep):
                top=np.min([n+nStep,self._Sensor['Pressure'].shape[0]])
                FSignal=mkl_fft.fft(self._Sensor['Pressure'][n:top,:],axis=1)
                k=np.floor(index[n:top]/(self._N1*self._N2)).astype(np.int64)
                j=index[n:top]%(self._N1*self._N2)
                i=j%self._N1
                j=np.floor(j/self._N1).astype(np.int64)
                FSignal=FSignal[:,IndSpectrum]
                if b3D==False:
                    assert(np.all(j==int(self._N2/2))) #all way up we specified the XZ plane at N2/2, this assert should pass
                pa= np.angle(FSignal)
                pp=np.abs(FSignal)
                
                self._PhaseMap[i,j,k]=pa
                self._PressMapFourier[i,j,k]=pp
                self._PressMapPeak[i,j,k]=self._Sensor['Pressure'][n:top,:].max(axis=1)
            self._InPeakValue=self._DictPeakValue['Pressure']
        else:
            FSignal=mkl_fft.fft(self._Sensor['Pressure'],axis=1)
            index=self._InputParam['IndexSensorMap'].astype(np.int64)
            k=np.floor(index/(self._N1*self._N2)).astype(np.int64)
            j=index%(self._N1*self._N2)
            i=j%self._N1
            j=np.floor(j/self._N1).astype(np.int64)
            FSignal=FSignal[:,IndSpectrum]
            if b3D==False:
                assert(np.all(j==int(self._N2/2))) #all way up we specified the XZ plane at N2/2, this assert should pass
            pa= np.angle(FSignal)
            pp=np.abs(FSignal)
            self._PhaseMap[i,k]=pa
            self._PressMapFourier[i,k]=pp
            self._PressMapPeak[i,k]=self._Sensor['Pressure'].max(axis=1)
            self._InPeakValue=self._DictPeakValue['Pressure'][:,int(self._N2/2),:]
        
        self._PressMapFourier*=2/self._Sensor['time'].size
        if b3D:
            print('Elapsed time doing phase and amp extraction from Fourier (s)',time.time()-t0)
        
        
    def PlotData(self):
  
        if self._bDisplay:
            plt.figure(figsize=(18,6))
            plt.subplot(1,4,1)
            plt.imshow(self._PressMapPeak.T/1e6,cmap=plt.cm.jet,
                       extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._FXDim.min(),self._FXDim.max())
            plt.ylim(self._FZDim.max(),self._FZDim.min())
            plt.colorbar()
            plt.title('Peak amp. (MPa)')
            plt.subplot(1,4,2)
            plt.imshow(self._PressMapFourier.T/1e6,cmap=plt.cm.jet,
                       extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._FXDim.min(),self._FXDim.max())
            plt.ylim(self._FZDim.max(),self._FZDim.min())
            plt.colorbar()
            plt.title('Fourier amp. (MPa)')
            plt.subplot(1,4,3)
            plt.imshow(self._InPeakValue.T/1e6,cmap=plt.cm.jet,
                       extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._FXDim.min(),self._FXDim.max())
            plt.ylim(self._FZDim.max(),self._FZDim.min())
            plt.colorbar()
            plt.title('InPeak amp. (MPa)')
            
            plt.subplot(1,4,4)
            plt.imshow(self._PhaseMap.T,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._FXDim.min(),self._FXDim.max())
            plt.ylim(self._FZDim.max(),self._FZDim.min())
            plt.colorbar()
            plt.title('Phase map (rad)')
            
        LineInPeak=self._InPeakValue[int(self._N1/2),:]/1e6
        LineFourierAmp=self._PressMapFourier[int(self._N1/2),:]/1e6
        
        if self._bDisplay:
            plt.figure(figsize=(12,8))
            plt.plot(self._ZDim,LineFourierAmp)
            plt.plot(self._ZDim,LineInPeak)
            plt.xlim(self._FZDim.min(),self._FZDim.max())
            plt.legend(['Fourier based','In-peak'])
        
        print('Peak pressure BabelFDTD (MPa) = %3.2f' %  (np.max(self._PressMapPeak)/1e6))
        
    def PlotData3D(self):
  
        if self._bDisplay:
            plt.figure(figsize=(18,6))
            plt.subplot(1,4,1)
            plt.imshow(self._PressMapPeak[:,int(self._N2/2),:].T/1e6,cmap=plt.cm.jet,
                       extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._FXDim.min(),self._FXDim.max())
            plt.ylim(self._FZDim.max(),self._FZDim.min())
            plt.colorbar()
            plt.title('Peak amp. (MPa)')
            plt.subplot(1,4,2)
            plt.imshow(self._PressMapFourier[:,int(self._N2/2),:].T/1e6,cmap=plt.cm.jet,
                       extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._FXDim.min(),self._FXDim.max())
            plt.ylim(self._FZDim.max(),self._FZDim.min())
            plt.colorbar()
            plt.title('Fourier amp. (MPa)')
            plt.subplot(1,4,3)
            plt.imshow(self._InPeakValue[:,int(self._N2/2),:].T/1e6,cmap=plt.cm.jet,
                       extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._FXDim.min(),self._FXDim.max())
            plt.ylim(self._FZDim.max(),self._FZDim.min())
            plt.title('InPeak amp. (MPa)')
            plt.colorbar()
            
        
            plt.subplot(1,4,4)
            plt.imshow(self._PhaseMap[:,int(self._N2/2),:].T,cmap=plt.cm.jet,extent=[self._XDim.min(),self._XDim.max(),self._ZDim.max(),self._ZDim.min()])
            plt.xlim(self._FXDim.min(),self._FXDim.max())
            plt.ylim(self._FZDim.max(),self._FZDim.min())
            plt.colorbar()
            plt.title('Phase map (rad)')
            

        LineInPeak=self._InPeakValue[int(self._N1/2),int(self._N2/2),:]/1e6
        LineFourierAmp=self._PressMapFourier[int(self._N1/2),int(self._N2/2),:]/1e6
        
        if self._bDisplay:
            plt.figure(figsize=(12,8))
            plt.plot(self._ZDim,LineFourierAmp)
            plt.plot(self._ZDim,LineInPeak)
            plt.xlim(self._FZDim.min(),self._FZDim.max())
            plt.legend(['Fourier based','In-peak'])
        
        print('Peak pressure BabelFDTD (MPa) = %3.2f' %  (np.max(self._PressMapPeak)/1e6))
      
    def ResamplingToFocusConditions(self,TESTNAME='test.h5',
                                    OutputPath='',
                                    bReload=False,
                                    bSkipSaveAndReturnData=False):
        if bReload: #use with caution, mostly if some modification to the export of an existing simulation
            DictPeakValue=ReadFromH5py('..'+os.sep+'DATA'+os.sep+'ALL-'+TESTNAME+'.h5')
            PhaseMap=DictPeakValue.pop('PhaseMap')
            PressMapFourier=DictPeakValue.pop('PressMapFourier')
            PressMapPeak=DictPeakValue.pop('PressMapPeak')
        else:
            DictPeakValue=self._DictPeakValue
            PhaseMap=self._PhaseMap
            PressMapFourier=self._PressMapFourier
            PressMapPeak=self._PressMapPeak
        
        InterAllData={}
        FRefX,FRefY=np.meshgrid(self._FXDim,self._FZDim)
        FStep=self._FZDim[1]-self._FZDim[0]
        
        for k in DictPeakValue:
            if k=='Pressure':
                Interpolator=interpolate.interp2d(self._XDim,self._ZDim,PressMapFourier.T)
            else:
                Interpolator=interpolate.interp2d(self._XDim,self._ZDim,DictPeakValue[k][:,int(self._N2/2),:].T)
            InterAllData[k]=Interpolator(self._FXDim,self._FZDim).reshape(FRefX.shape)
            
        #For phase data, it is better to use nearest interpolator to avoid the phase wrap artifacts
        Xfo,Zfo=np.meshgrid(self._XDim,self._ZDim)
        InterpPhase=interpolate.NearestNDInterpolator(list(zip(Xfo.flatten(),Zfo.flatten())),PhaseMap.T.flatten())
        Xf,Zf=np.meshgrid(self._FXDim,self._FZDim)
        pInterpphase=InterpPhase(Xf.flatten(),Zf.flatten())
        pInterpphase=np.reshape(pInterpphase,FRefX.shape)
        
        if self._bDisplay:
            plt.figure(figsize=(12,6))
            plt.subplot(1,2,1)
            plt.imshow(InterAllData['Pressure']/1e6,cmap=plt.cm.jet,
                       extent=[self._FXDim.min(),self._FXDim.max(),self._FZDim.max(),self._FZDim.min()])
            plt.colorbar()
            plt.title('amplitude (MPa)')
            
            plt.subplot(1,2,2)
            plt.imshow(pInterpphase,cmap=plt.cm.jet,extent=[self._FXDim.min(),self._FXDim.max(),self._FZDim.max(),self._FZDim.min()])
            plt.colorbar()
            plt.title('phase(rad)')
        n=1
        if self._bDisplay:
            plt.figure(figsize=(16,8))
            for k in InterAllData:
                if k=='Pressure':
                    continue
                if len(InterAllData)>=7:
                    plt.subplot(3,3,n)
                else:
                    plt.subplot(2,3,n)
                n+=1
                plt.imshow(InterAllData[k],cmap=plt.cm.jet,extent=[self._FXDim.min(),self._FXDim.max(),self._FZDim.max(),self._FZDim.min()])
                plt.colorbar()
                plt.title(k)
            
            
        print('Peak pressure BabelFDTD (MPa) = %3.2f' %  (np.max(InterAllData['Pressure'])/1e6))
        
        DataToSave={}
        DataToSave['p_amp']=InterAllData['Pressure'].T
        DataToSave['p_phase']=pInterpphase.T
        for k in InterAllData:
            if k == 'Pressure':
                continue
            DataToSave[k]=InterAllData[k].T

        DataToSave['x_vec']=self._FXDim
        DataToSave['y_vec']=self._FZDim
        DataToSave['N1']=self._N1
        DataToSave['N2']=self._N2
        DataToSave['N3']=self._N3
        DataToSave['SpatialStep']=self._SpatialStep
        DataToSave['PMLThickness']=self._PMLThickness
        DataToSave['AdjustedCFL']=self._AdjustedCFL
        if not(self._GMapTotal is None):
            DataToSave['GMapTotal']=self._GMapTotal
        if  bSkipSaveAndReturnData==False:
            SaveToH5py(DataToSave,os.path.join(OutputPath,TESTNAME+'.h5'))
        else:
            return DataToSave
        
        
    def ResamplingToFocusConditions3D(self,TESTNAME='test.h5',
                                    OutputPath='',
                                    bReload=False,
                                    bSkipSaveAndReturnData=False,
                                    bUseCupyToInterpolate=True):
        
        if bReload: #use with caution, mostly if some modification to the export of an existing simulation
            DictPeakValue=ReadFromH5py('..'+os.sep+'DATA'+os.sep+'ALL-'+TESTNAME+'.h5')
            PhaseMap=DictPeakValue.pop('PhaseMap')
            PressMapFourier=DictPeakValue.pop('PressMapFourier')
            PressMapPeak=DictPeakValue.pop('PressMapPeak')
        else:
            DictPeakValue=self._DictPeakValue
            PhaseMap=self._PhaseMap
            PressMapFourier=self._PressMapFourier
            PressMapPeak=self._PressMapPeak
            
        t0=time.time()
        if bUseCupyToInterpolate:
            RatioPPW=np.round((self._FXDim[1]-self._FXDim[0])/(self._XDim[1]-self._XDim[0]),5)
            gFXDim=(cupy.arange(self._FXDim.shape[0])*RatioPPW+self._PMLThickness).astype(np.float32)
            gFYDim=(cupy.arange(self._FYDim.shape[0])*RatioPPW+self._PMLThickness).astype(np.float32)
            gFZDim=(cupy.arange(self._FZDim.shape[0])*RatioPPW+self._PMLThickness+self._PaddingForKArray).astype(np.float32)
            FRefY,FRefX,FRefZ=cupy.meshgrid(gFYDim,gFXDim,gFZDim)
            gPressMapFourier=cupy.asarray(PressMapFourier)
            gPhaseMap=cupy.asarray(PhaseMap)
            gFPressure=cndimage.map_coordinates(gPressMapFourier, cupy.array([FRefX.ravel(),FRefY.ravel(),FRefZ.ravel()]))
            gFPressure=gFPressure.reshape(FRefX.shape)
            FPressure=gFPressure.get()
            gpInterpphase=cndimage.map_coordinates(gPhaseMap, cupy.array([FRefX.ravel(),FRefY.ravel(),FRefZ.ravel()]),order=0)
            gpInterpphase=gpInterpphase.reshape(FRefX.shape)
            pInterpphase=gpInterpphase.get()
            
        else:
            FRefY,FRefX,FRefZ=np.meshgrid(self._FYDim,self._FXDim,self._FZDim)
            orig=(self._XDim,self._YDim,self._ZDim)
            Interpolator=interpolate.RegularGridInterpolator(orig,PressMapFourier, 
                                                             fill_value=0,method='linear')
            FPressure=Interpolator(np.asarray((FRefX.ravel(),FRefY.ravel(),FRefZ.ravel())).T)
            FPressure=FPressure.reshape(FRefX.shape)
            
            Interpolator=interpolate.RegularGridInterpolator(orig,PhaseMap, 
                                                             fill_value=0,method='linear')
            pInterpphase=Interpolator(np.asarray((FRefX.ravel(),FRefY.ravel(),FRefZ.ravel())).T)
            pInterpphase=pInterpphase.reshape(FRefX.shape)
            
        t1=time.time()
        print('Elapsed time to interpolate results',t1-t0)
        
        if self._bDisplay:
            CfY=np.argmin(np.abs(self._FYDim))
            
            plt.figure(figsize=(12,6))
            plt.subplot(1,2,1)
            plt.imshow(FPressure[:,CfY,:].T/1e6,cmap=plt.cm.jet,
                       extent=[self._FXDim.min(),self._FXDim.max(),self._FZDim.max(),self._FZDim.min()])
            plt.colorbar()
            plt.title('Amplitude XZ (MPa)')
                        
            CfX=np.argmin(np.abs(self._FXDim))
            
            plt.subplot(1,2,2)
            plt.imshow(FPressure[CfX,:,:].T/1e6,cmap=plt.cm.jet,
                       extent=[self._FYDim.min(),self._FYDim.max(),self._FZDim.max(),self._FZDim.min()])
            plt.colorbar()
            plt.title('Amplitude YZ (MPa)')
            
            CfY=np.argmin(np.abs(self._FYDim))
            
            plt.figure(figsize=(12,6))
            plt.subplot(1,2,1)
            plt.imshow(pInterpphase[:,CfY,:].T,cmap=plt.cm.jet,
                       extent=[self._FXDim.min(),self._FXDim.max(),self._FZDim.max(),self._FZDim.min()])
            plt.colorbar()
            plt.title('Phase XZ (rad)')
                        
            CfX=np.argmin(np.abs(self._FXDim))
            
            plt.subplot(1,2,2)
            plt.imshow(pInterpphase[CfX,:,:].T,cmap=plt.cm.jet,
                       extent=[self._FYDim.min(),self._FYDim.max(),self._FZDim.max(),self._FZDim.min()])
            plt.colorbar()
            plt.title('Phase YZ (rad)')
           
        
        DataToSave={}
        DataToSave['p_amp']=FPressure.T
        DataToSave['p_phase']=pInterpphase.T
        DataToSave['x_vec']=self._FZDim
        DataToSave['y_vec']=self._FYDim
        DataToSave['z_vec']=self._FXDim
        DataToSave['N1']=self._N1
        DataToSave['N2']=self._N2
        DataToSave['N3']=self._N3
        DataToSave['SpatialStep']=self._SpatialStep
        DataToSave['PMLThickness']=self._PMLThickness
        DataToSave['AdjustedCFL']=self._AdjustedCFL
        if not(self._GMapTotal is None):
            DataToSave['GMapTotal']=self._GMapTotal
        if  bSkipSaveAndReturnData==False:
            SaveToH5py(DataToSave,os.path.join(OutputPath,TESTNAME+'.h5'))
        else:
            return DataToSave
        
