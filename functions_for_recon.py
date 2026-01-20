# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:11:31 2023
@author: thompson.3962
"""
import numpy as np 
import os
from PIL import Image
import cv2
from skimage import io
import os
import glob
import matplotlib.pyplot as plt
import pickle 
import shelve
import math
import tifffile as tf
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fft, ifftn


def fnGenericOTF(fnpath,isOTF):
    #wl = wl;
    #NA = NA;
    #Pxy = Pxy;
    cfg_1 = 'Huang2018';
    cfg_2 = 'emul';
    cfg_3 = 'expTestStack_1.00-1.70px';
    sigmaPSF=np.linspace(1.00,1.70,8);
    if isOTF == 0:
        NA = 1.4;
        Pxy = 65;
        Npx = 512;
        fn0 = '%iOTF_%.02fNA_%inm_%ipx_%s.tif',488,1.49,80,Npx,cfg_1;
        deconv_otfname = (fnpath, fn0);
    elif  isOTF==1:
        if 0:
            NA = 1.4;
            Pxy = 65;
            Npx = 512;
            fn0 = '488OTF_1.49NA_80nm_512px_%s.tif',488,1.49,80,Npx,cfg_2;
            deconv_otfname = os.path.join(fnpath, fn0);
        else:
            NA = 1.49;
            Pxy = 80;
            Npx = 512;
            fn0 = '%iOTF_%.02fNA_%inm_%ipx_%s.tif',488,1.49,80,Npx,cfg_2;
            deconv_otfname = os.path.join(fnpath, fn0)    
    elif  isOTF==2:
        NA = 1.49;
        Pxy = 80;
        Npx = 512;
        fn0 = '488OTF_1.49NA_80nm_512px_expTestStack_1.00-1.70px.tif';
        deconv_otfname = os.path.join(fnpath, fn0);
        return deconv_otfname, sigmaPSF


def imreadstack(imname):
    info = io.imread(imname);
    #num_images = len(info);
    zdepth, swidth, sheight = info.shape 
    f=np.zeros([zdepth,sheight,swidth]);
    for k in range(zdepth):
        f[k,:,:] =  info[k,:, :];
    f = np.swapaxes(f,0,1)
    f = np.swapaxes(f,1,2)
    return f
    
def column(matrix, i):
    return [row[i] for row in matrix]

def reconStartFolders(pathnameOut):
    qwer = pathnameOut + '/Pseudo-TIRF';
    os.makedirs(qwer, mode = 0o666);
    qwer = pathnameOut + '/SIM-Wiener';
    os.makedirs(qwer, mode = 0o666);
    qwer = pathnameOut + '/dispParams_01';
    os.makedirs(qwer);
    
def findFileNumber(fn):
    ix1 = fn.rfind('_')
    ix2 = fn.rfind('.')
    label = fn[ix1+1:ix2]
    nmLen = len(label)
    nm = int(label)
    return nm, nmLen
    
def genFN(fn,  isFolder = False, FN = './'):
    PWD = os.getcwd()   # returns current working directory
    os.chdir(FN)        # changes current working directory to FN (default is the same)
    if not FN.endswith(os.sep):
        FN += os.sep    # checks if FN ends with \ and if it doesn't, well now it does
    if isFolder:
        fn = fn + '_01' # adds _01 to fn
        fnSrch = glob.glob(FN + fn + '*')   # searches all files in FN that start with fn and returns a list of file paths
        if len(fnSrch) == 0:
            ix = 0
    while os.path.exists(FN + fn):
            ss = fn.split('_')
            ix = int(ss[-1]) + 1
            ixTxt = f'{ix:02}'      # formats ix as a 2 digit number with a leading 0
            fn = fn[:-2] + ixTxt
    else:
        # searches files starting with fn except last 4 characters that end with the last 4 characters of fn in FN
        # os.path.join makes sure  the directory path is formatted correctly (/ or \ depending on the system)
        # glob.glob returns list of file paths that match, recursive is to also search subdirectories
        fnSrch = glob.glob(os.path.join(FN, f"{fn[:-4]}*{fn[-4:]}"), recursive=True)
        if len(fnSrch) == 0:
            ix = 0
        else:
            ix, _ = findFileNumber(fnSrch[-1].name) # returns the number and length, ex: reconParamWiener_002.tif returns (2,3)
        ixTxt = f'_{ix+1:02d}'  # string that is the new file number formatted to two digits with leading zeros
        fn = fn[:len(fn)-4] + ixTxt + fn[len(fn)-4:]    # replaces fn _ix with _ix+1
        os.chdir(PWD)   # changes current wording directory to PWD
    return fn

def save(filename, *args):
    # Get global dictionary
    glob = globals()
    d = {}
    for v in args:
        # Copy over desired values
        d[v] = glob[v]
    with open(filename, 'wb') as f:
        # Put them in the file 
        pickle.dump(d, f)

def load(filename):
    # Get global dictionary
    glob = globals()
    with open(filename, 'rb') as f:
        for k, v in pickle.load(f).items():
            # Set each global variable to the value from the file
            glob[k] = v

def save(filename, *args):
    # Get global dictionary
    glob = globals()
    d = {}
    for v in args:
        # Copy over desired values
        d[v] = glob[v];
    with open(filename, 'wb') as f:
        # Put them in the file 
        pickle.dump(d, f)

def load(filename):
    # Get global dictionary
    glob = globals()
    with open(filename, 'rb') as f:
        for k, v in pickle.load(f).items():
            # Set each global variable to the value from the file
            glob[k] = v

    
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
    
def imageSizeSet(D,n): # comeback and finish
    #D: image or #px, n: # px output
    # D2: output image,  v_h: indices of image
    #input image size
    if np.size(D) == 1:
        K_h = D;
    else:
        K_h = [D.shape[0], D.shape[1]]; 
    N_h = [n,n];
    L_h = math.ceil((N_h-K_h) / 2);
    v_h = colonvec(L_h+1, L_h+K_h);
    D2=np.zeros(N_h);
    for i in range(1,np.size(D,3)):
        D2[tuple(v_h), i] = D[:, :, i];
        
def myimreadstack_TIRF(filename, frm1, nImg, testreadx, testready):
    # IMPORTANT###  nImg based on how many images are in your file!!!!
    nImg = 9
    with tf.TiffFile(filename) as t:  # Opens TIFF file to read and closes it after the loop
        img_read = np.zeros((nImg, testready, testreadx), dtype=np.uint16) # Unsigned 16-bit is to make sure values are correct
        for k in range(nImg):
            img_read[None, None, k] = t.pages[0+k].asarray() # Reads kth page and converts it to numpy array
    return img_read


def DOphase(rp, D, M, na, nphase):
    # na : nangles
    # np : nphases
    # M : inv_phase_matrix
    # D : dataFT
    # rp = 0
    import numpy as np 
    if np.shape(M) != (3, 3, 3):
        M = np.tile(M, (3, 1, 1))
        DMjk = np.zeros([512, 512, nphase], dtype=np.complex128)
        #rp = np.zeros((n, n, nphases*nangles))
        rp = rp = rp.astype('complex128')
    # calculate filtered data
    for i in range(na):  # theta
        for j in range(nphase):  # phase
                jj = ((i) * nphase) + j;
                for k in range(nphase):  # phase
                    kk = ((i) *  nphase) + k
                    #print(kk)
                    DMjk[:,:,k] = M[i][j][k] * D[:,:, kk]
                    rp[:,:, jj] = rp[:,:, jj] + DMjk[:,:,k]#come back to because it is not working in terms of 
    return rp
    
"""
DMjk = np.zeros([512, 512, nphases], dtype=np.complex128)
rp = np.zeros((n, n, nphases*nangles))
rp = rp = rp.astype('complex128')
for i in range(nangles):  # theta
    for j in range(nphases):  # phase
            jj = ((i) * nphases) + j;
            #print(jj);
            for k in range(nphases):  # phase
                kk = ((i) *  nphases) + k
                DMjk[:,:,k] = (M[i,j,k]) * dataFT[:,:, kk]
                rp[:,:, jj] = rp[:,:, jj] + DMjk[:,:,k]
                #M[0][1][1]
"""
def colonvec(m, M):
    # Takes two lists n and M and returns a list of lists v
    # where v[k] is a list of integers ranging from m[k] to M[k]
    n = len(m)
    N = len(M)
    K = max(n, N)
    v = [None] * K
    if n == 1:
        m = [m[0]] * K
    elif N == 1:
        M = [M[0]] * K
    for k in range(K):
        v[k] = list(range(m[k], M[k] + 1))
    return v

def dft(h, g):
    #convolution
    K = np.shape(h);
    L = np.shape(g);
    KL = tuple(map(sum, zip(K, L)))
    #KL = np.add(K,L);
    f = np.zeros(KL);# I dont believe this does anything
    #res1 = np. asarray(KL) # goes from tuple to array 
    #f = np.fft.fftn(H2, KL) * np.fft.fftn(H2flip, KL)
    #f = np.fft.ifftn(f, KL)
    # hw_45 = np.pad(lordwhyH, [(256, ), (256, )], mode='constant')
    #hw_88 = np.pad(H2flip, [(256, ), (256, )], mode='constant')
    #hw_89 = np.pad(H2, [(256, ), (256, )], mode='constant')
    f = np.fft.ifftn((np.fft.fftn(h, KL) * np.fft.fftn(g, KL)),KL);
    rret = range(0,(K[0]+L[0]-1));
    rret1 = range(0,(K[1]+L[1]-1));
    f = f[rret];
    f = f[:, rret1];
    #f_tmp4 = f[0:(K[0]+L[0]-1),0:(K[1]+L[1]-1)] #this works as well. 
    return f
    

def diffOrderOverlap(shftKernel,OTFmask,OTF,DO1,DO0):
    #calculates overlap of orders
    #OTF: OTF
    #OTFmask: OTF mask
    #shftKernel: freq domain shift kernel
    #1 (OTF MASK shift)
    OTF_MASKshifted = fftshift(fft2(ifftshift(fftshift(ifft2(ifftshift(OTFmask))) * shftKernel))); #OTF MASK shift
    OTF_MASKshifted[np.abs(OTF_MASKshifted)>0.9]=1;
    OTF_MASKshifted[np.abs(OTF_MASKshifted)!=1]=0;
    
    #2 (OTF  shift)
    OTFshifted = fftshift(fft2(ifftshift(fftshift(ifft2(ifftshift(OTF))) * shftKernel[:,:]))); #OTF shift 
    OTFshifted= OTFshifted * OTF_MASKshifted; #shifted masked OTF
    
    #3 (dfrctOrder1 shift)
    DO1shifted = fftshift(fft2(ifftshift(fftshift(ifft2(ifftshift(DO1))) * shftKernel[:,:]))); #dfrctOrder1 shift
    
    #4 = #1*#2*#3
    DO1overlap = DO1shifted * OTF_MASKshifted * OTF; #masked dfrctOrder1 shift prop
    
    # test#1
    DO0overlap = DO0 * OTFshifted;
    DO0_DO1overlap=np.conj(DO1overlap) * DO0overlap; #dfrctOrder0 leak prop dfrctOrder1 interference 
    overlap=np.abs(np.sum(DO0_DO1overlap[:]));
    
    # test#2
    overlapAreaImg = OTFmask * OTF_MASKshifted; #dfrctOrder0 leak area
    overlapArea=np.sum(overlapAreaImg[:]);
    
    overlap= overlap / overlapArea;
    return overlap
    
    
def bl_resize(original_img, new_h, new_w):
    # you run it by "bl_resize(otf, 512, 512)"
	#get dimensions of original image
	old_h, old_w = original_img.shape
	#create an array of the desired shape. 
	#We will fill-in the values later.
	resized = np.zeros((new_h, new_w))
	#Calculate horizontal and vertical scaling factor
	w_scale_factor = (old_w ) / (new_w ) if new_h != 0 else 0
	h_scale_factor = (old_h ) / (new_h ) if new_w != 0 else 0
	for i in range(new_h):
		for j in range(new_w):
			#map the coordinates back to the original image
			x = i * h_scale_factor
			y = j * w_scale_factor
			#calculate the coordinate values for 4 surrounding pixels.
			x_floor = math.floor(x)
			x_ceil = min( old_h - 1, math.ceil(x))
			y_floor = math.floor(y)
			y_ceil = min(old_w - 1, math.ceil(y))

			if (x_ceil == x_floor) and (y_ceil == y_floor):
				q = original_img[int(x), int(y)]
			elif (x_ceil == x_floor):
				q1 = original_img[int(x), int(y_floor)]
				q2 = original_img[int(x), int(y_ceil)]
				q = q1 * (y_ceil - y) + q2 * (y - y_floor)
			elif (y_ceil == y_floor):
				q1 = original_img[int(x_floor), int(y)]
				q2 = original_img[int(x_ceil), int(y)]
				q = (q1 * (x_ceil - x)) + (q2	 * (x - x_floor))
			else:
				v1 = original_img[x_floor, y_floor]
				v2 = original_img[x_ceil, y_floor]
				v3 = original_img[x_floor, y_ceil]
				v4 = original_img[x_ceil, y_ceil]

				q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
				q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
				q = q1 * (y_ceil - y) + q2 * (y - y_floor)

			resized[i,j] = q
	return resized.astype(np.uint8)



def calcShift(x, y):
    # Calculates the magnitude of SIM vector shift

    x = np.array(x).reshape(3, 3)   # Reshapes x and y from 9 column vector to 3x3 matrix
    x = x.T
    y = np.array(y).reshape(3, 3)
    y = y.T 
    x2 = x - x[0]    # Sets first row to 0s to apparently make calculations easier
    y2 = y - y[0]
    
    r = np.sqrt(x2**2 + y2**2)  # Finds distance
    r = r[1:,:]   
    r = r.T
    r = r.reshape(-1,1)
    return r

def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def regress_R1(OTFshifted_OTFoverlap_ang,DO1overlap6_ang,DO0overlap6_ang):
    from sklearn import linear_model 
    regr = linear_model.LinearRegression()
    R2 = np.zeros((1048576, 3),dtype = 'complex_')
    R2_a_ang=OTFshifted_OTFoverlap_ang;
    R2_b_ang = np.ravel(R2_a_ang, order='F').reshape(-1)
    R2_b_ang[R2_b_ang!=0]=1;
    R2[:,0] = R2_b_ang
    
    R2_c=DO1overlap6_ang;
    R2_d = np.ravel(R2_c, order='F').reshape(-1)
    R2[:,1] = R2_d
    #R2_b_ang[:,2]=R2_d;
    
    R2_c=DO0overlap6_ang;
    R2_d = np.ravel(R2_c, order='F').reshape(-1)
    R2[:,2] = R2_d
    #R2_b_ang[:,3]=R2_d;
    
    a = [R2[:,1]==0]
    a = a[0]

    a = R2[a,:]
    
    R2[[R2[:,1]==0],:]=[];
    R2_y=R2[:,1]; 
    R2_x=R2[:,2]; 
    [B,bint,r,rint,stats]=regr.fit(R2_y,R2_x);
    R2_angles=stats[1];
    dbgRegress = R2_angles;
    return dbgRegress;


"""    
    Junk for now: 
        #filewithsavevars='C:/Users/thompson.3962/Documents/Recon/shelve.out'
        def savelocalvaraibles(filewithsavevars): 
            my_shelf = shelve.open(filewithsavevars,'n') # 'n' for new
            for key in dir():
                try:
                    my_shelf[key] = globals()[key];
                except TypeError:
                        print('ERROR shelving')
            my_shelf.close()

        file = open("car.txt", "w")
        with open(r'C:/Users/thompson.3962/Documents/Recon/car.txt', 'w') as fp:
            for item in dir():
                # write each item on a new line
                fp.write("%s\n" % item)
            print('Done')

        def loadlocalvariables(filewithsavevars):
            my_shelf = shelve.open(filewithsavevars);
            for key in my_shelf:
                globals()[key]=my_shelf[key]
            my_shelf.close()
            
            #np.save(pathnameOut + filename[:-4] + '_diffShftx.npy', diffShftx = diffShftx)	
            #np.save(pathnameOut + filename[:-4] + '_diffShfty.npy', diffShfty = diffShfty)
        
    """
    
"""
# this is for the wienerweighparam
        if loadSIMPattern:  # Loads illumination pattern specified by 'illumFN'
            fd = ffr.myimreadstack_TIRF(illumFN, 1, 9, sx, sy)  # Reads 9-frame TIRF, returns result as array
            fd = swap9frmOrder(fd, orderSwapVector) # Swaps frame order based on orderSwapVector (which I think is the normal 1:9)
        else:
            if starframe + (nphases * nangles) * frmAvg - 1 > zstack:
                warnings.warn('The number of averaged frames should be smaller', Warning)
                return
            else:
                
    if bgflag == 1:
        bg = ffr.imreadstack(fn)    # Loads bgname image stack
        bckgrnd = bg[(end//2)+1-(sx)//2:(end//2)+(sx)//2, (end//2)+1-(sy)//2:(end//2)+(sy)//2]
    elif bgflag == 0:
        bckgrnd = np.zeros((sx, sy))
    """