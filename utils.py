import numpy as np
from PIL import Image
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
import hydra
from PIL import Image
import numpy as np
from skimage.segmentation import mark_boundaries
from scipy.ndimage.morphology import binary_dilation
import cv2

matplotlib.use("Agg")
os.environ["PATH"] = os.environ["PATH"] + ":/Library/TeX/texbin"  # for tex in matplotlib
plt.rc("font", family="serif")
plt.rc("text", usetex=True)


def get_dataset(dataset_class, **kwargs):
    """returns _target_ as defined in config"""
    return hydra.utils.instantiate(dataset_class, **kwargs)

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
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
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def plot_and_save_heatmap_only(heat_array, save_path):
    heat_array = heat_array.astype("float")
    h, w = heat_array.shape
    dpi = 100
    # normalize = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=False)
    colormap = plt.get_cmap("PuRd")
    fig = plt.Figure(figsize=(w / dpi, h / dpi), dpi=dpi, frameon=True)

    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    hm = ax.imshow(heat_array, cmap=colormap, interpolation="none")

    canvas.draw()
    heat_im = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape((h, w, 3))
    plt.colorbar(heat_im)
    plt.close()
    Image.fromarray(heat_im.astype("uint8")).save(save_path)

def plot_and_save_heatmap(arr, save_path):

    orig_cmap = plt.get_cmap("PuRd")
    shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.05, name="shifted")

    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 2), dpi=400)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    ax1.grid(False)
    ax1.tick_params(axis="both", which="both", length=0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    heat1 = ax1.imshow(arr, cmap=shifted_cmap, interpolation="None", vmin=0, vmax=np.max(arr))
    plt.tight_layout()
    fig.colorbar(mappable=heat1, cax=cax1, format="%.0f")
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close()


# Divide int into two closest integers
def int_div(n):
    if n % 2 == 0:
        return (int(n / 2), int(n / 2))
    else:
        return (int(n // 2), int(n // 2 + 1))

def compute_heatmap(loader, resolutions, ids, save_name):

    h, w = np.split(np.array(resolutions), 2, axis=1)
    w_max = w.max()
    h_max = h.max()
    heatmap = np.zeros((h_max, w_max))
    for _, gt in loader:
        gt = np.squeeze(gt.numpy())
        gt[~(np.isin(gt, ids))] = 0
        gt[np.isin(gt, ids)] = 1

        h_gap = h_max - gt.shape[0]
        w_gap = w_max - gt.shape[1]
        h_gap_div = int_div(h_gap)
        w_gap_div = int_div(w_gap)
        gt = np.pad(gt, (h_gap_div, w_gap_div), "constant", constant_values=(0))
        heatmap += gt
    plot_and_save_heatmap(heatmap, save_name)
    print("Heatmap saved: ", save_name, "Max. pixel value: ", np.max(heatmap))

def compute_heatmap_coda(loader, resolutions, ids, save_name):

    h, w = np.split(np.array(resolutions), 2, axis=1)
    w_max = w.max()
    h_max = h.max()
    heatmap = np.zeros((h_max, w_max))
    for gt in loader:
        gt = np.squeeze(gt.numpy())
        gt[~(np.isin(gt, ids))] = 0
        h_gap = h_max - gt.shape[0]
        w_gap = w_max - gt.shape[1]
        h_gap_div = int_div(h_gap)
        w_gap_div = int_div(w_gap)
        gt = np.pad(gt, (h_gap_div, w_gap_div), "constant", constant_values=(0))
        heatmap += gt
    plot_and_save_heatmap(heatmap, save_name)
    print("Heatmap saved: ", save_name, "Max. pixel value: ", np.max(heatmap))

def get_boundary_mask(arr, index=1, color=[0,1,0]):
    arr2 = np.copy(arr)
    arr2[arr != index] = 0
    arr2[arr == index] = 255
    gt = Image.fromarray(arr2.astype("uint8"))
    bd = mark_boundaries(gt, arr2)
    mask = binary_dilation(np.all(bd == [1, 1, 0], axis=-1).astype(int)).astype("uint8")
    mask = cv2.dilate(mask, kernel = np.ones((4, 4), np.uint8), iterations=1)
    bd[np.all(bd == np.ones(3), axis=-1)] = np.zeros(3)
    bd[mask == 1] = color
    im = (bd * 255).astype("uint8")
    return im

def compute_and_save_overlay_i(image, gt, ids, save_name):
    color = 255 * np.ones(image.shape)
    color[gt==255] = (0, 0, 0)
    color[np.isin(gt, ids)] = (255, 102, 0)
    out = np.copy(image)
    out[~np.all(color == 255*np.ones(3), axis=-1)] = \
          0.3*out[~np.all(color == 255*np.ones(3), axis=-1)] \
        + 0.7*color[~np.all(color == 255*np.ones(3), axis=-1)]
    overlay = np.copy(gt)

    blend = get_boundary_mask(overlay, index=0, color=[1,0,0])
    for i in ids:
        blend = blend + get_boundary_mask(overlay, index=i, color=[-1,1,0])

    out[~np.all(blend == np.zeros(3), axis=-1)] = blend[~np.all(blend == np.zeros(3), axis=-1)]
    Image.fromarray(out.astype("uint8")).save(save_name)

def compute_and_save_overlays(data, loader, save_dir):
    for i, (image, gt) in enumerate(loader):
        save_name = data.images[i].split('/')[-1].split('.')[0] + '_blend.png'
        save_path = os.path.join(save_dir, save_name)
        gt = np.array(gt)
        gt = np.squeeze(gt)
        image = np.squeeze(image)
        compute_and_save_overlay_i(image, gt, data.ood_id, save_path)
