import h5py
from astropy import units
from astropy import constants as c

from . import scale as s
from . import dalt  as d
import numpy as np
from rapplot import * 

 
def image2d(image, stokes_ind, data_id):
    """Create 2D array of one of the Stokes parameters for plotting."""

    n_box = len(image[data_id[stokes_ind]])              #depends on camera pixles you chose for GRRT   
    n_pixel_per_box = len(image[data_id[stokes_ind]][0]) #100
    pixels = int(np.sqrt(n_pixel_per_box))               # pixels per block side 10x10 

    n_blocks_side = int(round(np.sqrt(n_box)))
    
    # Total image dimensions
    width = n_blocks_side * pixels
    height = (n_box // n_blocks_side) * pixels

    image_array = np.zeros((height, width), dtype=float)

    for i in range(n_box):
        block_row = i // n_blocks_side
        block_col = i % n_blocks_side
        block_data = np.array(image[data_id[stokes_ind]][i])
        array = np.reshape(block_data, (pixels, pixels))
        image_array[
            block_row*pixels:(block_row+1)*pixels,
            block_col*pixels:(block_col+1)*pixels
        ] = array
        

    return image_array

def load_hdf5(f, halfrange, snapshot, mask, resize, **kwargs):

    data_id = list(f.keys())

    MBH = (4.14e6 * units.M_sun).to(units.M_sun, equivalencies=s.GR)
    dist = 8.127 * units.kpc
    freq = 230 * units.GHz
    rg = (c.G * MBH /c.c**2 )
    Tunit = rg/c.c
    mas = (rg/dist)* 206264.806*1000.
    time = (int(snapshot) *10. *Tunit).to(units.s, equivalencies=s.GR)    
    
    stokes_ind = 0
        
    if resize: 
        img = image2d(f, stokes_ind, data_id)
      
    elif isinstance(f, h5py.File):  
        img = f[data_id[stokes_ind]][:]

            
    if mask is not None:
        img = img.copy()
        img[mask] = 0

    width, height = halfrange*2, halfrange*2
    return d.Image(img, MBH, dist, freq, time, width, height, **kwargs)



def load_img(f, ind, halfrange, mask, resize, **kwargs):
    if isinstance(f, h5py.File):
        return load_hdf5(f, halfrange, ind, mask=mask, resize=resize, **kwargs)
    with h5py.File(f, "r") as g:
        return load_hdf5(g, halfrange, ind, mask=mask, resize=resize, **kwargs)

def load_summ(f, **kwargs):
    with h5py.File(f, "r") as h:
        Mdot  = h['Mdot'][()]
        Ladv  = h['Ladv'][()]
        nuLnu = h['nuLnu'][()]
        Ftot  = h['Ftot'][()]
        img   = load_img(h, **kwargs)
    return Mdot, Ladv, nuLnu, Ftot, img

def load_mov(fs, snapshots, halfrange, mean=False, mask=None, resize=False, **kwargs):
    if isinstance(fs, str):
        fs = [fs]
        
    times = []
    imgs  = [] # collect arrays in list and then cast to np.array() in
               # d.Image() all at once is faster than concatenate
    for f, snapshot in zip( fs, snapshots ):
        img = load_img(f, snapshot, halfrange, mask=mask, resize=resize, **kwargs)
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
