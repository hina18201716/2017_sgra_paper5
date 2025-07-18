import h5py
from astropy import units
from astropy import constants as c

from . import scale as s
from . import dalt  as d
import numpy as np

def load_hdf5(f, snapshot, **kwargs):

    def get(u, k):
        return u[k][()]
    
    # images = h5py.File(f,'r')
    keys = [key for key in f.keys()]
    
    # tauI = f['tau1.000000e+12'][()]

    #Note that no flips or transposes have been made.  This may need to occur in analysis scripts.
    MBH = (4.14e6 * units.M_sun).to(units.M_sun, equivalencies=s.GR)
    dist = 8.127 * units.kpc
    freq = 230 * units.GHz
    
    rg = (c.G * MBH /c.c**2 )
    Tunit = rg/c.c
#     TODO : double check
    time = (int(snapshot) *10. *Tunit).to(units.s, equivalencies=s.GR)    
    
    strokes_ind = 0
    for i in range(0,len(f[keys[strokes_ind]])):
        width = int(np.sqrt(len(f[keys[strokes_ind]][i])))
        img    = ((np.reshape(f[keys[strokes_ind]][i],(width,width))))
    height = width
    # print(MBH, dist, freq, time, width, height)
    return d.Image(img, MBH, dist, freq, time, width, height, **kwargs)

def load_img(f, ind, **kwargs):
    if isinstance(f, h5py.File):
        return load_hdf5(f, ind, **kwargs)
    with h5py.File(f, "r") as g:
        return load_hdf5(g, ind, **kwargs)

def load_summ(f, **kwargs):
    with h5py.File(f, "r") as h:
        Mdot  = h['Mdot'][()]
        Ladv  = h['Ladv'][()]
        nuLnu = h['nuLnu'][()]
        Ftot  = h['Ftot'][()]
        img   = load_img(h, **kwargs)
    return Mdot, Ladv, nuLnu, Ftot, img

def load_mov(fs, snapshots, mean=False, **kwargs):
    if isinstance(fs, str):
        fs = [fs]
        
    times = []
    imgs  = [] # collect arrays in list and then cast to np.array() in
               # d.Image() all at once is faster than concatenate
    for f, snapshot in zip( fs, snapshots ):
        img = load_img(f, snapshot, **kwargs)
        times.append(img.meta.time)
        imgs.append(img)

    meta = img.meta
    meta.time = units.Quantity(times)

    #from scipy import ndimage
    #import numpy as np
    #imgs = [ndimage.rotate(im, 140, reshape=False) for im in imgs]

    if mean:
        import numpy as np
        imgs = np.mean(imgs, axis=0)

    return d.Image(imgs, meta=meta)
