from matplotlib.path import Path 

import numpy  as np
import pandas as pd
import h5py
from pathlib import Path

from astropy    import units as u
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
# matplotlib.use('Agg')

from scipy.interpolate import RegularGridInterpolator
from scipy.signal      import argrelextrema

from common import hallmark as hm
from common import mockservation as mk
from common import io_raptor as io
from common import dalt
from common import viz

import plotscripts
from plotscripts import subtract, image2d, loadfigure, getimagedata

import rapplot


#Computing relevant constants
M=5e6 * rapplot.MSUN
d=8.127 * rapplot.KPC

rg = (rapplot.G*M/rapplot.SPEED_OF_LIGHT**2.)

mas = (rg/d)* rapplot.MAS_IN_DEG

Tunit =rg/rapplot.SPEED_OF_LIGHT

halfrange=20 #in rg
# ====================================================

def import_ab(a, i, cond):
    file = f'output/shadow_a{a:.2f}_i{i:g}.h5' # change this to your local folder
    with h5py.File(file) as h:
      alpha = h['a'][:]
      beta = h['b'][:]

    if cond:
      index_apos = np.where(alpha > 0)[0]
      aright = index_apos[beta[index_apos].argmin()]

      for bb in np.linspace(-10e-3, 10e-3, 10):
        alpha = np.append(alpha, alpha[aright])
        beta = np.append(beta, bb)

    return (alpha, beta)
# ====================================================



def plotregion(imgs, ax):
 
    intensity = np.sum(imgs, axis=0)
    dominant_index = np.argmax(imgs, axis=0)
    dominant_index[intensity <=1e-16 ] = -1  
  
    cmap = plt.get_cmap('tab10')
    colors = [cmap(9)] + [cmap(i) for i in range(3)] 
    color_map = np.array(colors)
    
    extent = [-halfrange * mas, halfrange * mas, -halfrange * mas, halfrange * mas]
    display_index = dominant_index + 1
    ax.imshow(display_index, origin='lower', cmap=plt.matplotlib.colors.ListedColormap(color_map),
              extent=extent)

    labels = ['Background', 'n0', 'n1', 'n2']
    handles = [mpatches.Patch(color=color_map[i], label=labels[i]) for i in range(4)]

    return handles


def plot_va(mov, ax, n, i ): 
    vis = mk.mockserve(mov, N=6000)

    U, V = vis.uvd
    u = np.linspace(0, U/2, num=vis.shape[-1])
    v = np.linspace(-V/2, V/2, num=vis.shape[-2], endpoint=False)
    t = vis.meta.time.value

    amp = RegularGridInterpolator((t, v, u[::-1]), abs(vis[..., ::-1]))
    phi = RegularGridInterpolator((t, v, u[::-1]), np.angle(vis[..., ::-1]))

    bmin = 6e9
    bmax = 20e10
    uvd = np.linspace(0, bmax, 1000)
    
    # i = 0
    phi_angle = np.pi * i / 180
    u = uvd * np.cos(phi_angle)
    v = uvd * np.sin(phi_angle)

    mask = u <= 0
    p = np.array([np.repeat(t[0], np.sum(mask)), v[mask], u[mask]]).T
    m = np.array([np.repeat(t[0], np.sum(~mask)), -v[~mask], -u[~mask]]).T

    s = np.zeros(len(uvd))
    s[mask] = amp(p)
    s[~mask] = amp(m)

    uvd_G = uvd / 1e9
    s_Jy = s * 1e-23

    if n == 100:
        line, = ax.semilogy(uvd_G, s_Jy, '-', linewidth=0.5, label='Total')
    else:
        line, = ax.semilogy(uvd_G, s_Jy, '-', linewidth=0.5, label=f'n={n}')

    return [line]
# ===================================================================================================

# under developement
# change to class
def f(phi, inc):
    
    #get critical curve 
    alpha1, beta1 = import_ab(1.01, inc, True) 
    
    # critical curve in r, phi
    R = np.sqrt(alpha1**2 + beta1**2)
    PHI = np.arctan2(beta1, alpha1) 
    alpha = np.round((alpha1 / pixel_scale + 0.5) * n_pix).astype(int)
    beta = np.round((beta1 / pixel_scale + 0.5) * n_pix).astype(int)
    
    # print(f'In shape {phi.shape}')
    R_ref = np.sqrt(alpha1**2 + beta1**2)
    
    phi       = np.mod(phi + np.pi, 2*np.pi) - np.pi
    PHI_ref   = np.mod(PHI + np.pi, 2*np.pi) - np.pi

    inds=[]
    for p in phi:
        dPHI = np.abs(PHI_ref - p)
        ind = np.nanargmin(dPHI) 
        inds.append(ind)
     
    inds = np.array(inds)
   
    phi_new = PHI_ref[inds]
    r_new = R_ref[inds]
    
    # print(f'Out PHI shape {phi_new.shape}')
    # print(f'Out R shape {r_new.shape}')

    return r_new, phi_new    

def setcoords(Img, inc): 
    xmin, xmax = 350, 700
    ymin, ymax = 350, 650
    Img_c  = Img[ymin:ymax, xmin:xmax]
    A_c    = A[ymin:ymax, xmin:xmax]
    B_c    = B[ymin:ymax, xmin:xmax]
    
    alpha_min = FoV * ((xmin + 0.5) / n_pix - 0.5)
    alpha_max = FoV * ((xmax - 0.5) / n_pix - 0.5)
    beta_min  = FoV * ((ymin + 0.5) / n_pix - 0.5)
    beta_max  = FoV * ((ymax - 0.5) / n_pix - 0.5)
    extent=[alpha_min, alpha_max, beta_min, beta_max]
    print(extent)
    print(Img_c.shape)

    r_pix = np.sqrt(( A_c**2 + B_c**2 ))
    phi_pix = np.arctan2(B_c,A_c)
    
    phi_flat = phi_pix.ravel()
    r_out, phi_out = f(phi_flat)
    
    r_crit = r_out.reshape(phi_pix.shape)
    phi_crit = phi_out.reshape(phi_pix.shape)
    

def findmask(Img_c, r_pix, r_crit, phi_pix, r_max=[0.97, 0.97], r_min=[0.8,0.8]):
    #mask r
    r_mask_1 = (r_pix < r_max[0] * r_crit) & (r_pix > r_min[0] * r_crit)
    r_mask_2 = (r_pix < r_max[1] * r_crit) & (r_pix > r_min[1] * r_crit)
    # r_mask = (r_pix < 4.2) & (r_pix > 3.2)
    
    #mask phi 
    phi_c = np.pi/2
    phi_mask_1 =  ( phi_pix > 0.93 *phi_c) & ( phi_pix < 1.05*phi_c )
    phi_mask_2 =  (-phi_pix > 0.93 *phi_c) & (-phi_pix < 1.08*phi_c )
    # phi_mask =  (np.abs(phi_pix) > 0.94*phi_c ) & (np.abs(phi_pix) < 1.13*phi_c )
    
    mask_good = (Img_c > 0.5e16) 
    mask_bad  = (~mask_good) & r_mask_1 & phi_mask_1 * r_mask_2 & phi_mask_2
    # mask_bad  = (~mask_good) & r_mask & phi_mask

    i_good, j_good = np.nonzero(mask_good)
    Img_good = Img_c[(i_good, j_good)]

    i_bad_1,  j_bad_1  = np.nonzero(mask_bad_1)
    Img_bad_1 = Img_c[(i_bad_1, j_bad_1)]

    i_bad_2,  j_bad_2  = np.nonzero(mask_bad_2)
    Img_bad_2 = Img_c[(i_bad_2, j_bad_2)]

    i_bad, j_bad = np.concatinate(i_bad_1, i_bad_2), np.concatinate(j_bad_1, j_bad_2)

    return Img_good, i_good, j_good, i_bad, j_bad, mask_bad

#expand boundary 
def expand(mask_bad, i_bad, j_bad): 
    dilated_bad  = binary_dilation(mask_bad, structure=np.ones((5, 5)))
    edge_mask = dilated_bad & ~mask_bad
    adjacent_values = Img_c[edge_mask]
    i_edge, j_edge = np.nonzero(edge_mask)
    
    coords_bad = np.column_stack((i_bad, j_bad))
    coords_edge = np.column_stack((i_edge, j_edge))
    coords_combined = np.vstack([coords_bad, coords_edge])
    coords_unique = np.unique(coords_combined, axis=0)
    
    i_bad, j_bad = coords_unique[:, 0], coords_unique[:, 1]
    
    Img_bad = Img[i_bad, j_bad]
    print(i_bad.shape, j_bad.shape)

    phi_bad = np.arctan2(b[i_bad], a[j_bad])  # double check convention
    r_bad   = np.sqrt(( a[j_bad] * a[j_bad] + b[i_bad] * b[i_bad] )) / f(phi_bad)[0]
    print(f"Img shape {Img_good.shape}")
    print(f"Good points shape {r_good.shape}, {phi_good.shape}")
    print(f"Bad points shape {r_bad.shape}, {phi_bad.shape}")
    return Img_bad, phi_bad, r_bad, i_bad, j_bad