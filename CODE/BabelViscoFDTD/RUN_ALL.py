'''
Main script for ITRUSST intercomparison effort with BabelViscoFDTD

https://github.com/ProteusMRIgHIFU/BabelViscoFDTD
Samuel Pichardo, Ph.D
Assistant Professor
Radiology and Clinical Neurosciences, Hotchkiss Brain Institute
Cumming School of Medicine,
University of Calgary
samuel.pichardo@ucalgary.ca
www.neurofus.ca

ITRUSST BabelViscoFDTD benchmarking

Global execution of all Notebooks for simulations

This script will generate a notebook for each benchmark with plots

'''
import glob
import os
import json
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor,CellExecutionError
import gc

LDir=[]
ListPPW=[6]#,[9,12] #12 PPW was the one selected for the paper, 9 PPW shows very close results while requiring less computing resources
for PPW  in ListPPW:
    for bm  in ['BM1','BM2','BM3','BM4','BM5','BM6','BM7','BM8']:
        for st in ['MP1','MP2']:
            SufPPW='_%iPPW' % PPW
            for dv in ['SC1','SC2']:
                LDir.append('PH1-'+bm+'-'+st+'-'+dv+'_BabelViscoFDTD'+SufPPW)
print(LDir)

#specify Path for output data and temporary files
OutputPath =os.path.join('..','..','DATA','BabelViscoFDTD','')
if not os.path.isdir(OutputPath):
    print('Creating output directory: '+OutputPath)
    os.makedirs(OutputPath)

RasterInputPath=os.path.join('..','..','intercomparison','skull-cartesian','')
if not os.path.isdir(RasterInputPath):
    raise SystemError('Directory for voxelized skull maps (from stl) does not exist [%s]' % (RasterInputPath))
#specify ID string for GPU and BackEnd
GPUName='A6000'

#set this to True to force recalculation of existing benchmarks
bForceRecalculate = True

for lName in LDir:
    #Uncomment this below to skip files already calculated
    datan = OutputPath+lName+'.h5'
    if os.path.isfile(datan) and bForceRecalculate==False:
        print('skipping',datan)
        continue
    
    print('*'*80+'\n'+lName+'\n'+'*'*80)
    with open('InputParam.json','w') as f:
        Input={}
        Input['OutputPath']=OutputPath
        Input['TESTNAME']=lName
        Input['RasterInputPath']=RasterInputPath
        Input['GPUName']=GPUName
        if ('BM7' in lName or 'BM8' in lName) and '12PPW' in lName:
            #for the biggest domains, a system with 256 GB or more is required and using only CPU
            #Please note that 9PPW produces a very close result and can be calculated with GPU with ~32 GB RAM
            COMPUTING_BACKEND=0 #0 for OpenMP, 1 for CUDA, 2 for OpenCL and 3 for Metal
        else:
            COMPUTING_BACKEND=1 #0 for OpenMP, 1 for CUDA, 2 for OpenCL and 3 for Metal
        Input['COMPUTING_BACKEND']=COMPUTING_BACKEND
        json.dump(Input,f)
    
    print('*'*80)
    print('+'*10 + ' RUNNING FOR '+lName + ' ' + '+'*10 )
    print('-'*80)
    gc.collect()
    with open('MASTER_NOTEBOOK_BABELVISCO.ipynb') as f:
        nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
    ep = ExecutePreprocessor(timeout=-1,kernel_name='python3')
    gc.collect()
    try:
        out=ep.preprocess(nb, {'metadata': {'path': './'}})
    except CellExecutionError:
        out = None
        msg = 'Error executing the notebook "%s".\n\n' % lName
        msg += 'See output notebook  for the traceback.'
        print(msg)
        raise
    finally:
        with open('Executed-'+lName+'.ipynb', 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
    