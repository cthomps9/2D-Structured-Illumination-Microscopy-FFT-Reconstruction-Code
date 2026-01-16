# -*- coding: utf-8 -*-
"""
This take care of the variables from the need to run the loadCFrecon and the 
taking of varaibles from cfgExp. 



@author: thompson.3962
"""
#Imports
import math as math
import numpy as np
import functions_for_recon as ffr
import os
import glob
#from PIL import Image
import cv2
import skimage
from skimage import io
import shutil
import shelve
import pickle
""" WienerShiftParam """
import warnings
from skimage.transform import resize
import numpy as np
import math
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import Functions_for_WienerShiftParam as ffwsp
import functions_for_recon as ffr

""" Intial Parameters to be set"""

"""Setup up Sys"""
cols = 2
rows = 3
sys = [[0 for i in range(cols)] for j in range(rows)]
sys[0][0] = 'wl';
sys[1][0] = 'NA';
sys[2][0] = 'Pxy';
sys[0][1] =  488;
sys[1][1] =  1.49;
sys[2][1] = 80;

#back to parameters
wlex = 488; #[nm] excitation
NA = 1.49;  
Pxy = 80; #[nm]
PSFsigma = 1; #[px] reference for PSFsigma1_airyRad (PSFsigma:1)
wlCoeff = 488/wlex;

isLoadReconParams = 0; # 0: default 1: skips parameter optimization (debug)
wienFilterCoeff = .001;
frmAvg = 1;
phaseShift = [1, 1, 1]; # SIM pattern phase shift [au, integers]
freqCutoff0 = 220*wlCoeff; #OTF mask radius [pxSIM]
"""
Check out the Huang et al. paper to confirm this. 
    #fcc = fp2/fp1 = (Pxy2/ NA2) /  (Pxy1/NA1) see calcFreqParams.ppt
    #fcc = 1; #default [Huang et.al.]
"""

fcc0 = 1.28*wlCoeff; #(Pxy/NA)/(65/1.4); % freqCutoff multiplier 1.15 ~= (80/1.49)/(65/1.4)
fco0 = 1.1564/wlCoeff; #1.1530 = 211/183 (freqPos ratios IDL/Huang)
fco0 = fcc0;
freqGcenter = 200*wlCoeff; #search center [pxSIM] (grating frequency)
isBGcorrection = 0;  #1: load system background image 0: no bg correction
isOTF = 2; # 0: Huang 1: emul 2: exp
muHessian = 150; # Hessian param1
sigmaHessian = 1; # Hessian param2
bgflag=isBGcorrection; #load system background
frmAvg0=frmAvg; #for recon parameter detection
isFreqMask = 1; #final mask (wienerCore) (default:1)
isSmoothPhaseDetection = 1; #phase angle fit and peak detection (wienerWeighParam)
copyParamsDIR = []; #copies recon params from another run (default:[])
#output settings -------------------------------------
isHessian = 0; 
testPSF = 0; #tries different PSF
runEmul = 0; #
#constants -------------------------------------
NpxOTF = 512;
NmPSF = 10; # px magnification\
""" paths to change are the following including pathnameOutOTF"""
imgFolder = 'C:/Users/thompson.3962/Documents/Recon/imgfolder/imgfolder/'
simFolder = imgFolder;
# folder below is full of OTF stuff
imgOTFfolder = 'C:/Users/thompson.3962/Documents/Recon/imgfolder/reconData/dataOTF/';
otfFolder = imgOTFfolder;
imgBGfolder = 'C:/Users/thompson.3962/Documents/Recon/imgfolder/imgBGfolder';
filename0 = 'acqimg.tif'; # Name of the file. 
"""end of mandatory path that should be changed for code to run"""
## Creates names Folder and files ============================================  
pathnameIn = imgFolder;
outFolder = simFolder;
otfFolder = imgOTFfolder; #OTF
bgFolder = imgBGfolder; #background 
outFolder = outFolder; #output
filenameList = filename0;
"""Determines the save function"""

def save(filename, *args):
    # Get global dictionary
    glober = globals()
    d = {}
    for v in args:
        # Copy over desired values
        d[v] = glober[v];
    with open(filename, 'wb') as f:
        # Put them in the file 
        pickle.dump(d, f)
        
""" Code param before WienerCall"""

#recon cfg
bgflag=isBGcorrection; #load system background
frmAvg0=frmAvg; #for recon parameter detection

# illumination 
rpRat=phaseShift; #SIM pattern phase shift [au, integers]
wienCoeff=2; #Wiener param1 (2: 3 rotations Hessian-SIM)
nphases = 3;
nangles = 3;
    
# frequency cutoff coeff (scaling) see calcFreqParams.ppt
freqCutoff0 = freqCutoff0*fcc0; #OTF mask radius [pxSIM]
freqGcenter = freqGcenter;    
freqGband=50; #half width of the band around 'freqGcenter' [px] (grating frequency)
    
# algo params
mu=muHessian; #Hessian param1
sigma=sigmaHessian; #Hessian param2
    
# output
isReconInterleaved=0;
    
#dbg
loadSIMPattern=0;
isSwapOPO = 0; 

# calculate
frmAvg = loadSIMPattern + frmAvg0; #loadSIMPattern=> frmAvg:1

# constants
twoPi=2*np.pi;
frmMul=1;

orderSwapVector = list(range(1, 10))	# No swap
    
#image size
cake = os.path.join(pathnameIn, filenameList)
info = io.imread(cake)
starframe0 = 1
zstack0, sx, sy = info.shape # Obtains the width and height from info
if frmAvg == 1:
    nEstimate = math.floor(zstack0 / 9); # estimates the number of images in groups of nine. 

###BG###
fn = os.path.join(bgFolder,'background','.tif'); # creates a tif file

###OTF###
OTFfn0,sigmaPSF = ffr.fnGenericOTF(otfFolder,isOTF);
otf0 = ffr.imreadstack(OTFfn0);
OTFfn0 = os.path.splitext(OTFfn0);	
OTFfn0 = OTFfn0[0];
otf0 = otf0.astype(np.uint16).astype(np.float64);
nOTF = np.shape(otf0)[0];
OTFfn0 = os.path.join(outFolder,'OTF_',fn+ OTFfn0,'.tif');

"""
        import pickle;
        filewithsavevars = 'store.pckl'
        def load(filename):
            # Get global dictionary
            glob = globals()
            with open(filename, 'rb') as f:
                for k, v in pickle.load(f).items():
                    # Set each global variable to the value from the file
                    glob[k] = v;
        ffr.load(filewithsavevars);
        """

nMovie = 1 # number of movies chosen
pathnameOutOTF = []
if nOTF>1:
    pathnameOutOTF = 'C:/Users/thompson.3962/Documents/Recon/imgfolder/correct order_OTF/';
    os.makedirs(pathnameOutOTF)
for j in range(4,6):
    otf = otf0[:,:,j];
    def WienerShiftParam(otf, rpRat, filename0, pathnameIn, pathnameOutParams, starframe, nphases, nangles, freqCutoff0, frmAvg, sx, sy, freqGcenter, freqGband, pathnameOut,fcc):
        dbg = 0;
        dataFN = pathnameIn + filename0;
        illumFN = pathnameIn + filename0 + "_patternSIM.tif";
        patternFreqFN = pathnameOutParams + filename0 + '_patternFreq.tif';
        imageFourierFN1 = pathnameOutParams + filename0 + '_imageFourier1.tif';
        imageFourierFN2 = pathnameOutParams + filename0 + '_imageFourier2.tif';
        fd = ffr.myimreadstack_TIRF(dataFN,starframe,(nphases * nangles) * frmAvg, sx, sy);
        # Check OTF and Data Size
        otf = resize(otf, (512, 512), order = 1, mode = 'reflect', anti_aliasing = True);
        H = otf;
        fdd = np.zeros((sx, sy, nphases * nangles));
        n = max(sx, sy, 512) 
        fc = fcc * math.ceil(220 * (n / 512))
        freqLim = math.ceil(freqGcenter * (n / 512))
        freqBand = math.ceil(freqGband * (n / 512))
        phase_matrix0 = [[0, 0, 0], [rpRat[0] / sum(rpRat), 1, -rpRat[0] / sum(rpRat)], [(rpRat[0] + rpRat[1]) / sum(rpRat), 1, -(rpRat[0] + rpRat[1]) / sum(rpRat)]]
        ret = np.ones((3,3), dtype = int);
        ret2 = ret * 1J *2*np.pi;
        ret3 = ret2 * phase_matrix0;
        phase_matrix = np.exp(ret3) 

        fdd = fd
        test = np.zeros((4, 1))
        diffShftx = np.zeros((nphases*nangles, 1))
        diffShftx[:, 0] = n
        diffShfty = np.zeros((nphases*nangles, 1))
        diffShfty[:, 0] = n
        DorrNorm6 = np.zeros((1023, 1023, 6))
        DOcorrNorm6 = np.zeros((2*n-1, 2*n-1, (nangles)*(nphases-1)))
        k_x, k_y = np.meshgrid(np.arange(-(n)//2, (n+1)//2), np.arange(-(n)//2, (n+1)//2))	# Creates n x n grid of spacial frequencies
        k_r = np.sqrt(k_x**2 + k_y**2)
        indi = k_r > fc
        H[indi] = 0
        H = np.abs(H)
        inv_phase_matrix = np.linalg.inv(phase_matrix)
        rp = np.zeros((n, n, nphases*nangles))
        rpt = np.zeros((2*n,2*n,nphases*nangles))
        fd512 = np.zeros((n, n, 9))
        K_h = [fd.shape[0], fd.shape[1]]
        N_h = (n, n)
        L_h = np.ceil((np.subtract(N_h, K_h))/2).astype(int)
        v_h = ffr.colonvec(L_h+1, L_h+K_h)
        hw = np.zeros((N_h))
        for ii in range(0, nphases*nangles):
            fd512[:, :, ii] = fd[ii, :,:];
        temer = ifftshift(fd512)
        temer1=fft2(np.swapaxes(np.swapaxes(temer,0,2),1,2))
        dataFT = fftshift(temer1) # the transforms arent working the same as matlab
        dataFT = np.swapaxes(np.swapaxes(dataFT,0,2),0,1)
        H1 = H
        H1[H1 != 0] = 1	# Creates mask where non-zero elements replaced by 1
        H2 = H1
        H9 = np.tile(H1, [1, 1, nphases*nangles])	# Copies the H1 filter accross the 9 stacks
        H9 = np.reshape(H9, (512,512,9), order="F")
        dataFT_2 = H9 * dataFT;
    	# H1
        K_h = H1.shape
        N_h = [i * 2 for i in K_h];
        L_h = np.ceil((np.subtract(N_h, K_h))/2).astype(int)
        v_h = ffr.colonvec(L_h+1, L_h+K_h)
        #hw = np.zeros((N_h))
        hw = np.pad(H1, [(256, ), (256, )], mode='constant')
        H1 = hw
        lordwhyH = otf; # gotta do it because python wipe out any idea of otf afterwards
        lordwhyH[indi] = 0
        lordwhyH = np.abs(lordwhyH)
        hw_45 = np.zeros((N_h))
        hw_45 = np.pad(lordwhyH, [(256, ), (256, )], mode='constant')
        lordwhyH = hw_45
        rp = ffr.DOphase(rp,dataFT_2,inv_phase_matrix,nangles,nphases)
        rp = rp / (np.abs(rp) + np.finfo(float).eps)
        for rpi in range(0, (nangles)*(nphases-1), 2): # this goes from 1 - 7
            errCoeffx = 1;
            errCoeffy = 1;
            rp = np.multiply(rp, H9);
            ix1 = math.ceil(rpi/2)*nphases + 1	#[1, 4, 7]
            ix2 = math.ceil(rpi/2)*nphases	#[0, 3, 6]
            qwer1 = DO0 = rp[:, :, ix1];
            qwer2 = DO1 = rp[:, :, ix2];

    		# Initial Estimate maxx and maxy
            H2flip = H2[::-1, ::-1]
            H2corr = ffr.dft(H2, H2flip)
            H2corrBW = H2corr
            H2corrBW[H2corrBW < 0.9] = 0
            H2corrBW[H2corrBW != 0] = 1

    		# Search Illum Pattern Freq Band Position
            qwer2_ = qwer2.conj()
            qwer2_ = qwer2_[::-1, ::-1] 
            DOcorr=ffr.dft(qwer1,qwer2_) 
            DOcorrMasked = np.multiply(DOcorr, H2corrBW)
            H2corr = ffr.dft(H2, H2flip) # have to rerun because python will wipe it out otherwise. 
            DOcorrNorm0 = np.abs(DOcorrMasked / (H2corr + np.finfo(float).eps)) 
            k_x, k_y = np.meshgrid(np.arange(-(2*n)//2+1, (2*n)//2), np.arange(-(2*n)//2+1, (2*n)//2))
            k_r = np.sqrt(k_x**2 + k_y**2)
            clip = ((k_r < (freqLim - freqBand)) | (k_r > (freqLim + freqBand)))
            DOcorrNorm = DOcorrNorm0
            DOcorrNorm[clip] = 0
            yy, xx = np.unravel_index(np.argmax(DOcorrNorm), DOcorrNorm.shape) 
            maxx = xx + 1 
            maxy = yy + 1 
            # Spatial Freq Vectors
            kx = twoPi * (maxx - n) / (2 * n)	
            ky = twoPi * (maxy - n) / (2 * n)	
            x = list(range(2*n))
            y = np.arange(2*n).reshape((2*n, 1))	
            xx2 = np.tile(x, (2*n,1))
            yy2 = np.tile(y, (1,2*n))

            #changes the size from 512 to 1024
            hw16 = np.pad(qwer2, [(256, ), (256, )], mode='constant')
            qwer2_2 = hw16
            hw17 =np.pad(qwer1, [(256, ), (256, )], mode='constant')
            qwer1_1 = hw17
            
            shftKernel = np.exp(1j*(kx*xx2 + ky*yy2));
            oo = ffr.diffOrderOverlap(shftKernel, H1, lordwhyH, qwer2_2, qwer1_1);
            maxx_tmp1 = maxx - 1e-5;
            maxx_tmp2 = maxx + 1e-5;
            maxy_tmp1 = maxy - 1e-5;
            maxy_tmp2 = maxy + 1e-5;
            kx_tmp1 = twoPi * (maxx_tmp1 - n) / (2 * n);	
            kx_tmp2 = twoPi * (maxx_tmp2 - n) / (2 * n);
            ky_tmp1 = twoPi * (maxy_tmp1 - n) / (2 * n);
            ky_tmp2 = twoPi * (maxy_tmp2 - n) / (2 * n);
            
            for ii in range(0, 4):	
                if ii == 0:
                    kxtest = kx_tmp1
                    kytest = ky
                elif ii == 1:
                    kxtest = kx_tmp2
                    kytest = ky
                elif ii == 2:
                    kxtest = kx
                    kytest = ky_tmp1
                else:
                    kxtest = kx
                    kytest = ky_tmp2
                shftKernel = np.exp(1j * (kxtest * xx2 + kytest * yy2))
                test[ii] = ffr.diffOrderOverlap(shftKernel, H1, lordwhyH, qwer2_2, qwer1_1);
            if test[0] > test[1]:
                flag_maxx = -1
            elif test[0] < test[1]:
                flag_maxx = 1
            else:
                warnings.warn('Can not estimate the pattern wave vector')
            if test[2] > test[3]:
                flag_maxy = -1
            elif test[2] < test[3]:
                flag_maxy = 1
            else:
                warnings.warn('Can not estimate the pattern wave vector')

            maxx_tmp = maxx
            maxy_tmp = maxy
            wix = 0
            while (errCoeffx > 1e-4) or (errCoeffy > 1e-4):
    			# Maxy
                maxy_tmp1 = maxy - 1e-5
                maxy_tmp2 = maxy + 1e-5
                ky_tmp1 = twoPi * (maxy_tmp1 - n) / (2 * n)
                ky_tmp2 = twoPi * (maxy_tmp2 - n) / (2 * n)
                for ii in range(2, 4):
                    if ii == 2:
                        kxtest = twoPi * (maxx - n) / (2 * n)
                        kytest = ky_tmp1
                    else:
                        kxtest = twoPi * (maxx - n) / (2 * n)
                        kytest = ky_tmp2
                    shftKernel = np.exp(1j * (kxtest * xx2 + kytest * yy2))
                    test[ii] = ffr.diffOrderOverlap(shftKernel, H1, lordwhyH, qwer2_2, qwer1_1)
                if test[2] > test[3]:
                    flag_maxy = -1
                elif test[2] < test[3]:
                    flag_maxy = 1
                else:
                    flag_maxy = -1 * flag_maxy
                while (errCoeffx > 1e-4):
                    maxy_tmp = maxy + flag_maxy * errCoeffx
                    kytest = twoPi * (maxy_tmp - n) / (2 * n)
                    kxtest = twoPi * (maxx - n) / (2 * n)
                    shftKernel = np.exp(1j * (kxtest * xx2 + kytest * yy2))
                    oo_tmp = ffr.diffOrderOverlap(shftKernel, H1, lordwhyH, qwer2_2, qwer1_1)
                    if oo_tmp <= oo:
                        errCoeffx = 0.5 * errCoeffx
                    else:
                        oo = oo_tmp
                        maxy = maxy_tmp
                        break
                MAXY = maxy;
                maxx_tmp1 = maxx - 1e-5
                maxx_tmp2 = maxx + 1e-5
                kx_tmp1 = twoPi * (maxx_tmp1 - n) / (2 * n)
                kx_tmp2 = twoPi * (maxx_tmp2 - n) / (2 * n)
                for ii in range(0, 2):
                    if ii == 0:
                        kxtest = kx_tmp1
                        kytest = twoPi * (maxy - n) / (2 * n)
                    else:
                        kxtest = kx_tmp2
                        kytest = twoPi * (maxy - n) / (2 * n)
                    shftKernel = np.exp(1j * (kxtest * xx2 + kytest * yy2))
                    test[ii] = ffr.diffOrderOverlap(shftKernel, H1, lordwhyH, qwer2_2, qwer1_1)
                if test[0] > test[1]:
                    flag_maxx = -1
                elif test[0] < test[1]:
                    flag_maxx = 1
                else:
                    flag_maxx = -1 * flag_maxx
                while (errCoeffy > 1e-4):
                    maxx_tmp = maxx + flag_maxx * errCoeffy
                    kytest = twoPi * (maxy - n) / (2 * n)
                    kxtest = twoPi * (maxx_tmp - n) / (2 * n)
                    shftKernel = np.exp(1j * (kxtest * xx2 + kytest * yy2))
                    oo_tmp = ffr.diffOrderOverlap(shftKernel, H1, lordwhyH, qwer2_2, qwer1_1)
                    if oo_tmp <= oo:
                        errCoeffy = 0.5 * errCoeffy
                    else:
                        oo = oo_tmp
                        maxx = maxx_tmp
                        break
                MAXX = maxx
                wix += 1
            diffShftx[1 + rpi + math.ceil((rpi-1)/2), :] = maxx
            diffShfty[1 + rpi + math.ceil((rpi-1)/2), :] = maxy
            diffShftx[2 + rpi + math.ceil((rpi-1)/2), :] = 2 * n - maxx
            diffShfty[2 + rpi + math.ceil((rpi-1)/2), :] = 2 * n - maxy
            DorrNorm6[:, :, rpi] = DOcorrNorm

    	# Save (get exact syntax)
        with open('diffShftx.npy', 'wb') as dsx:
            np.save(dsx, diffShftx)
        with open('diffShfty.npy', 'wb') as dsy:
            np.save(dsy, diffShfty)
        with open('DorrNorm6.npy', 'wb') as dn6:
            np.save(dn6, DorrNorm6)
        with open('dataFN.npy', 'wb') as dFN:
            np.save(dFN, dataFN)
        with open('inv_phase_matrix.npy', 'wb') as IPM:
            np.save(IPM, inv_phase_matrix)
        return diffShftx, diffShfty, DorrNorm6, inv_phase_matrix;

    
    PSFsgm = sigmaPSF[j];
    fco = fco0 / PSFsgm;
    fcc = fcc0;
    fco = fco0;
    if nOTF > 1:
        pathnamejr = "0%ssignal%d" % (j, PSFsgm);
        pathnameOut0 = os.path.join(pathnameOutOTF, pathnamejr);
        OTFfn0_ = 'OTF_488OTF_1.49NA_80nm_512px_expTestStack_1.00-1.70px';
        OTFjr = '%s' % (j);
        OTFfn = os.path.join(pathnameOutOTF, OTFfn0_, OTFjr, '.tif');
    else:
        OTFfn = OTFfn0
        pathnameOut0 = outFolder
        cv2.imwrite(OTFfn, otf.astype('uint16'))
    for i in range(1, nMovie + 1):
        pathnameOut = pathnameOut0;
        cfgFN2 = pathnameOut;
        ffr.reconStartFolders(pathnameOut); 

    # Wiener Call begins 
    filename = filenameList;
    if frmAvg == 1:
        pathnameOutParams = ffr.genFN('dispParams',1,pathnameOut); 
        pathnameOutParams = pathnameOut + pathnameOutParams + '/';
    for ix in range(1, nEstimate + 1):
        filewithsavevars = 'store.pckl'
        save(filewithsavevars, 'otf', 'NA', 'NmPSF','NpxOTF','OTFfn','OTFfn0','OTFfn0_','OTFjr','PSFsgm','PSFsigma','Pxy','bgFolder','bgflag','cake', 'cfgFN2','cols','copyParamsDIR','fcc','fcc0','fco','fco0','filename','filename0','filenameList','filewithsavevars','fn','freqCutoff0','freqGband','freqGcenter','frmAvg','frmAvg0','frmMul','imgBGfolder','imgFolder','imgOTFfolder','info','isBGcorrection','isFreqMask','isHessian','isLoadReconParams','isOTF','isReconInterleaved','isSmoothPhaseDetection','isSwapOPO','ix','loadSIMPattern','mu','muHessian','nEstimate','nMovie','nOTF','nangles','nphases','orderSwapVector','otf','otf0','otfFolder','outFolder','pathnameIn','pathnameOut','pathnameOut0','pathnameOutOTF','pathnameOutParams','pathnamejr','phaseShift','rows','rpRat','runEmul','sigma','sigmaHessian','sigmaPSF','simFolder','starframe0','sx','sy','testPSF','twoPi','wienCoeff','wienFilterCoeff','wlCoeff','wlex','zstack0');
        starframe = starframe0 + (ix - 1) * 9;
        zstack = zstack0;
        WienerShiftParam(otf, rpRat, filename0, pathnameIn, pathnameOutParams, starframe, nphases, nangles, freqCutoff0, frmAvg, sx, sy, freqGcenter, freqGband, pathnameOut,fcc);
        with open('diffShftx.npy', 'rb') as dsx:
            diffShftx = np.load('diffShftx.npy');
        with open('diffShfty.npy', 'rb') as dsy:
            diffShfty = np.load('diffShfty.npy');
        with open('DorrNorm6.npy', 'rb') as dn6:
            DorrNorm6 = np.load('DorrNorm6.npy');
        with open('dataFN.npy', 'rb') as dFN:
            dataFN_numbers = np.load('dataFN.npy');
        with open('inv_phase_matrix.npy', 'rb') as IPM:
            inv_phase_matrix = np.load('inv_phase_matrix.npy');

        WienerCore();
fnrecon = 're-' + filename0 + '.tif';