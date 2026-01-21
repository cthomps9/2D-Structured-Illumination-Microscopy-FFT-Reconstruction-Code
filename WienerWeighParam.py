# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:15:49 2023
@author: thompson.3962
Code for WienerWeighParam just in matlab terms 
"""

import math
import pickle
#from calcShift import calcShift
import cv2
from scipy import ndimage
from scipy.signal import savgol_filter
#from myimreadstack_TIRF import myimreadstack_TIRF
import warnings
#from colonvec import colonvec
#from DOphase import DOphase
#from swap9frmOrder import swap9frmOrder
#from regress_R2 import regress_R2
import numpy as np
import functions_for_recon as ffr
import math
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fft, ifftn
from PyEMD import EMD_D
EMD_D = EMD_d()
import emd 




# Functions I have not defined yet
from emd import emd # This is super long and I don't want to do this right now

def WienerWeighParam(sx, fn, wlex, dataFN, diffShftx, diffShfty, bgflag, fco, sy, otf, illumFN, orderSwapVector, starframe, nphases, nangles, frmAvg, zstack, loadSIMPattern, inv_phase_matrix):
    
    # OTF Frequency Cuttoff and Sampling
    n_512 = max([512, sx, sy])
    fcANG = 120
    if wlex == 488:
        fcABS = 105 
    elif wlex == 561:
        fcABS = 120
    elif wlex == 647:
        fcABS = 80
    fc_con = math.ceil(fco * fcABS * (n_512 / 512))
    fc_ang = math.ceil(fco * fcANG * (n_512/512))
    dPhase = 0.02   # Sampling distance (phase)
    dAmp = 0.02     # Sampling distance (amplitude)
    nphases = 3
    # Stores new x and y shifts scaled based on n_512 by taking average of shftx and shfty, basically accounts for change in image size
    diffShftx_512 = diffShftx * (n_512 // math.floor((diffShftx[0,0]+diffShfty[0,0])/2))
    diffShfty_512 = diffShfty * (n_512 // math.floor((diffShftx[0,0]+diffShfty[0,0])/2))
    shiftMag = ffr.calcShift(diffShftx_512, diffShfty_512)
    sMinMax = np.array([np.min(shiftMag), np.max(shiftMag)])    # Returns 1x2 matrix array([min, max]) of shiftMag
    overlapSize = shiftMag - 2*np.array([fcANG, fcABS])
    
    # H1  H9 (OTF) -- don't ask me what H1 and H9 means, it's probably some mathematical notation
    H_ang = otf
    H_ang = cv2.resize(H_ang, (n_512, n_512), interpolation=cv2.INTER_LINEAR)   # Bilinear interpolation
    H_con = otf
    H_con = cv2.resize(H_ang, (n_512, n_512), interpolation=cv2.INTER_LINEAR)   # Bilinear interpolation
    k_x, k_y = np.meshgrid(np.arange(-(n_512)//2, (n_512)//2), np.arange(-(n_512)//2, (n_512)//2))  # Creates coordinate grid n_512 x n_512 centered at origin
    k_r = np.sqrt(k_x**2 + k_y**2)  # Computes magnitude of the frequency values at each point in the meshgrid
    indi_ang = k_r > fc_ang     # Creates boolean mask for values in Fourier transform with frequencies greater than fc_angle
    indi_con = k_r > fc_con     # Same deal, but for values in the Fourier transform with frequencies greater than fc_con
    H_ang[indi_ang] = 0 # All values greater than fc_ang are 0
    H_con[indi_con] = 0 # Value greater than fc_con are 0
    H_ang = np.abs(H_ang)   # Takes absolute value of Fourier transform
    H_con = np.abs(H_con)   # which is important since transform can have negative values
    H1_ang = H_ang
    H1_con = H_con
    H1_ang[H1_ang != 0] = 1 # All non-zero values are 1
    H1_con[H1_con != 0] = 1 # which creates a binary mask for the OTF
# ... put it here for the wienerweighparam which is in functions for recon
    fd = ffr.myimreadstack_TIRF(dataFN, starframe, (nphases * nangles) * frmAvg, sx, sy)
    # pointless# fd = swap9frmOrder(fd, orderSwapVector)
    #fdd_an = np.zeros((sx, sy, 9))
    #for frm in range(0, nphases*nangles, 1):
        # This averages together chunks of frmAvg on consecutive frames to reduce noise, I think
    #    fdd_an[:, :, frm] = np.sum(fd[:, :, frm:(nphases*nangles):(nphases*nangles)*frmAvg], axis=2) / frmAvg
    #fd = fdd_an
    fd = (np.swapaxes(np.swapaxes(fd,0,2),0,1))
    fd512 = np.zeros((n_512, n_512, nphases*nangles))
    K_h = [fd.shape[0], fd.shape[1]]
    N_h = [n_512, n_512]
    L_h = np.ceil((np.array(N_h) - np.array(K_h)) / 2).astype(int)
    v_h = ffr.colonvec(L_h+1, L_h+K_h)
    hw = np.zeros(N_h)

    bckgrnd = np.zeros((sx, sy))
    for ii in range(nphases*nangles):
        fd[:, :, ii] = fd[:, :, ii] - bckgrnd   # This subtracts background
        fd512[:, :, ii] = fd[:, :, ii]

    # Fourier Transform
    lemer = ifftshift(fd512)
    lemer1=fft2(np.swapaxes(np.swapaxes(lemer,0,2),1,2))
    dataFT = fftshift(lemer1) # the transforms arent working the same as matlab
    dataFT = np.swapaxes(np.swapaxes(dataFT,0,2),0,1)

    # Initialize
    amp6 = np.zeros((3 * (nphases - 1), 1))
    angle6 = np.zeros((3 * (nphases - 1), 1))
    OTFshifted_OTFoverlap_con = np.zeros((2 * n_512, 2 * n_512, (nangles) * (nphases - 1)),dtype = 'complex_')
    OTFshifted_OTFoverlap_ang = np.zeros((2 * n_512, 2 * n_512, (nangles) * (nphases - 1)), dtype = 'complex_')
    DO1overlapNorm_con = np.zeros((2 * n_512, 2 * n_512, (nangles) * (nphases - 1)), dtype = 'complex_')
    DO1overlapNorm_ang = np.zeros((2 * n_512, 2 * n_512, (nangles) * (nphases - 1)), dtype = 'complex_')
    DO1overlap6_ang = np.zeros((2 * n_512, 2 * n_512, (nangles) * (nphases - 1)), dtype = 'complex_')
    DO0overlap6_ang = np.zeros((2 * n_512, 2 * n_512, (nangles) * (nphases - 1)), dtype = 'complex_')
    rp = np.zeros((n_512, n_512, nphases * nangles))
    temp_separated = np.zeros((n_512, n_512, nphases))
    x = np.arange(2 * n_512)
    y = np.arange(2 * n_512)[:, np.newaxis]
    xx2 = np.tile(x, (2 * n_512, 1))
    yy2 = np.tile(y, (1, 2 * n_512))

    # High Resolution OTF (512 --> 1024)
    K_h = np.array(H1_ang.shape)
    N_h = 2 * K_h
    L_h = np.ceil((N_h - K_h) / 2).astype(int)
    v_h = ffr.colonvec(L_h + 1, L_h + K_h)
    hw = np.zeros(N_h)
    
    H_ang = otf
    H_ang = cv2.resize(H_ang, (n_512, n_512), interpolation=cv2.INTER_LINEAR)   # Bilinear interpolation
    H_con = otf
    H_con = cv2.resize(H_ang, (n_512, n_512), interpolation=cv2.INTER_LINEAR)   # Bilinear interpolation
    H_ang[indi_ang] = 0 # All values greater than fc_ang are 0
    H_con[indi_con] = 0 # Value greater than fc_con are 0
    H_ang = np.abs(H_ang)   # Takes absolute value of Fourier transform
    H_con = np.abs(H_con)   
    
    hw_H_ang = np.pad(H_ang, [(256, ), (256, )], mode='constant')
    hw_H_con = np.pad(H_con, [(256, ), (256, )], mode='constant')
    hw_H1_ang = np.pad(H1_ang, [(256, ), (256, )], mode='constant')
    hw_H1_con = np.pad(H1_con, [(256, ), (256, )], mode='constant')

    
    # Diffraction Orders (DO) Phase Correction
    rp = ffr.DOphase(rp, dataFT, inv_phase_matrix, nangles, nphases)
    OTF_ang = hw_H_ang
    OTF_con = hw_H_con
    OTFmask_ang = hw_H1_ang
    OTFmask_con = hw_H1_con
    for rpi in range(1, (nangles)*(nphases-1)+1, 1):
        DiffOrder0 = rp[:,:,int(np.ceil(rpi/2))*3-2]
        DiffOrder1 = rp[:,:,int(np.ceil(rpi/2))+2*int(np.floor(rpi/2))-1]
        hw_DiffOrder0 = np.pad(DiffOrder0, [(256, ), (256, )], mode='constant')        
        DiffOrder0 = hw_DiffOrder0
        hw_DiffOrder1 = np.pad(DiffOrder1, [(256, ), (256, )], mode='constant')
        DiffOrder1 = hw_DiffOrder1
        kxtest = 2*np.pi*(diffShftx_512[rpi+1+int(np.floor((rpi-1)/2))-1]-n_512)/(2*n_512)
        kytest = 2*np.pi*(diffShfty_512[rpi+1+int(np.floor((rpi-1)/2))-1]-n_512)/(2*n_512)
        shftKernel = np.exp(1j*(kxtest*xx2+kytest*yy2))
        # 1 OTF Mask Shift -- computes shifted OTFs for the angle and contrast channels, or something like that idk
        OTF_MASKshifted_ang = fftshift(fft2(ifftshift(fftshift(ifft2(ifftshift(OTFmask_ang)))*shftKernel)))
        OTF_MASKshifted_con = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(OTFmask_con)))*shftKernel[:,:])))
        OTF_MASKshifted_ang[np.abs(OTF_MASKshifted_ang)>0.9] = 1
        OTF_MASKshifted_ang[np.abs(OTF_MASKshifted_ang)!=1] = 0 # Creates binary mask that masks out any high-frequency noise in the OTF
        OTF_MASKshifted_con[np.abs(OTF_MASKshifted_con)>0.9] = 1
        OTF_MASKshifted_con[np.abs(OTF_MASKshifted_con)!=1] = 0
        # 2 Shifted Masked OTF -- I think this shifts the OTF of the angle and contrast channel and applies the mask
        OTFshifted_ang = fftshift(fft2(ifftshift(fftshift(ifft2(ifftshift(OTF_ang)))*shftKernel)))
        OTFshifted_con = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(OTF_con)))*shftKernel)))
        OTFshifted_ang *= OTF_MASKshifted_ang
        OTFshifted_con *= OTF_MASKshifted_con
        # 3 Diffraction Order 1 Shift
        DO1shifted = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(DiffOrder1)))*shftKernel)))
        # 4 Masked DO1 Shift Prop
        DO1overlap_ang = DO1shifted*OTF_MASKshifted_ang*OTF_ang
        DO1overlap_con = DO1shifted*OTF_MASKshifted_con*OTF_con
        # 5 Masked DO1 Shift Prop
        DO0overlap_ang = DiffOrder0*OTFshifted_ang
        DO0overlap_con = DiffOrder0*OTFshifted_con
        OTFshifted_OTFoverlap_ang[:,:,rpi-1] =OTFshifted_ang*OTF_ang  # used as a mask
        OTFshifted_OTFoverlap_con[:,:,rpi-1] = OTFshifted_con*OTF_con  # used as a mask
        DO1overlapNorm_ang[:,:,rpi-1] = DO1overlap_ang/(DO0overlap_ang+np.finfo(float).eps)  # the data to be masked
        DO1overlapNorm_con[:,:,rpi-1] = DO1overlap_con/(DO0overlap_con+np.finfo(float).eps)  # the data to be masked
        DO1overlap6_ang[:,:,rpi-1] = DO1overlap_ang  # only for regression
        DO0overlap6_ang[:,:,rpi-1] = DO0overlap_ang  # only for regression

    #oDO1ang = np.angle(DO1overlapNorm_ang) #summary looks crazy 
    aqwer_1 = np.real(DO1overlapNorm_ang)
    aqwer_2 = np.imag(DO1overlapNorm_ang)
    oDO1ang = np.arctan2(aqwer_2, aqwer_1)
    oDO1abs = np.abs(DO1overlapNorm_con)
    xPhase = np.arange(-np.pi, np.pi + dPhase, dPhase) # column not a row like in matlab
    xAmp = np.arange(0, 0.6 + dAmp, dAmp) # column not a row like in matlab
    
    # Phase and Amplitude of Diffraction Patterns
    for ii in range(1, (nangles)*(nphases)):
        c = oDO1ang[:, :, ii-1]
        cc = oDO1abs[:, :, ii-1]
        d = np.ravel(c, order='F').reshape(-1)
        dd = np.ravel(cc, order='F').reshape(-1)
        
        a_ang = OTFshifted_OTFoverlap_ang[:, :, ii-1] # Larger mask for oDO1ang
        a_con = OTFshifted_OTFoverlap_con[:, :, ii-1] # Smaller mask for oDO1abs
        b_ang = np.ravel(a_ang, order='F').reshape(-1)
        b_con = np.ravel(a_con, order='F').reshape(-1)
        b_ang[b_ang!=0] = 1
        b_con[b_con!=0] = 1
        b_ang = np.column_stack((b_ang, d))
        b_con = np.column_stack((b_con, dd))
        b_ang = b_ang[b_ang[:, 0]!=0]
        b_con = b_con[b_con[:, 0]!=0]
        e = b_ang[:, 1]
        ee = b_con[:, 1]

        # histogram oDO1
        oDO1angMASKED=e.conj().T;
        oDO1absMASKED=ee.conj().T;
        gHist=np.histogram(oDO1angMASKED,xPhase); #oDO1ang
        ggHist=np.histogram(oDO1absMASKED,xAmp); #oDO1abs slightly short 1*29 not 30
        # Works perfectly up to here 04/28

# maybe something like this 
        gHist_1 = gHist[0]
        gHist_1 = np.array(gHist_1)
        ggHist_1 = ggHist[0]
        ggHist_1 = np.array(ggHist_1)

        # Smoothing
        imf = emd(gHist_1)  
        imf = emd.sift.sift(gHist_1)# Empirical Mode Decomposition
        imf2 = emd(ggHist_1)
        imf2 = emd.sift.sift(ggHist_1)
        isSmoothPhaseDetection = 1
        if isSmoothPhaseDetection:  # default: 0
            g = np.sum(imf[0:, :], axis=1)
            g = ffr.smooth(g, 5)
            #if np.count_nonzero(g == np.max(g)) > 1:
            #    pass    # This seems to be something that used to exist but doesn't anymore but I don't feel like deleting it
        #elif np.max(gHist) < 50:
        #    g = np.sum(imf[4:, :], axis=0)
        #else:
            #g = np.sum(imf[3:, :], axis=0)
        gg = np.sum(imf2[0:, :], axis=1)
        # maximize(1) This maxes the window of a plot but we're not doing that
        p = g
        pp = gg
        h = (np.where(g == np.max(g))[0]) - 1
        hh = (np.where(gg == np.max(gg))[0]) + 1
        angle6[ii-1] = -np.pi + np.mean(h) * dPhase
        amp6[ii-1] = np.mean(hh) * dAmp
        dbgRegress = ffr.regress_R1(OTFshifted_OTFoverlap_ang[:,:,ii-1], DO1overlap6_ang[:,:,ii-1], DO0overlap6_ang[:,:,ii-1])
        with open(pathnameOut + filename[:-4] + '_angleWeigh.pickle', 'wb') as f:
            pickle.dump({'angle6': angle6, 'amp6': amp6}, f)