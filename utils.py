import numpy as np
from PIL import Image
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
import hydra

matplotlib.use('Agg')
os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'  # for tex in matplotlib
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

def get_dataset(dataset_class, **kwargs):
    """returns _target_ as defined in config"""
    return hydra.utils.instantiate(dataset_class, **kwargs)


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    function taken from
    https://stackoverflow.com/questions/7404116/...
        ...defining-the-midpoint-of-a-colormap-in-matplotlib
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {  'red': [],  'green': [], 'blue': [],  'alpha': []  }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap




def plot_and_save_heatmap_only(heat_array, save_path):
    heat_array = heat_array.astype('float')
    h, w = heat_array.shape
    dpi = 100
    #normalize = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=False)
    colormap = plt.get_cmap('PuRd')
    fig = plt.Figure(figsize=(w/dpi, h/dpi), dpi=dpi, frameon=True)
    
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    hm = ax.imshow(heat_array, cmap=colormap, interpolation='none')
    
    canvas.draw()
    heat_im = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape((h, w, 3))
    plt.colorbar(heat_im)
    plt.close()
    Image.fromarray(heat_im.astype("uint8")).save(save_path)

def plot_and_save_heatmap(arr, save_path):

    orig_cmap= plt.get_cmap('PuRd')
    shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.05, name='shifted')

    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 2), dpi=400)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    ax1.grid(False)
    ax1.tick_params(axis='both', which='both', length=0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    heat1 = ax1.imshow(arr, cmap=shifted_cmap, interpolation='None', vmin=0, vmax=np.max(arr))
    plt.tight_layout()
    fig.colorbar(mappable=heat1, cax=cax1, format="%.0f")
    fig.savefig(save_path, bbox_inches='tight', transparent=True)
    plt.close()


def compute_heatmap(loader, resolutions, ids, save_name):

    h, w = np.split(np.array(resolutions), 2, axis=1)
    w_max = w.max()
    h_max = h.max()
    heatmap = np.zeros((h_max, w_max))
    for gt in loader:
        gt = np.squeeze(gt.numpy())
        gt[~(np.isin(gt, ids))] = 0
        gt[np.isin(gt, ids)] = 1
        h_gap = h_max - gt.shape[0] 
        w_gap = w_max - gt.shape[1]
        gt = np.pad(gt, ((int(h_gap/2), int(h_gap/2)), (int(w_gap/2), int(w_gap/2))), 'constant', constant_values=(0))
        heatmap += gt
    plot_and_save_heatmap(heatmap, save_name)
    print("Heatmap saved: ", save_name, "Max. pixel value: ", np.max(heatmap))
       


