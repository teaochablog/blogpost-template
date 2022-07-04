import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from cycler import cycler
from io import BytesIO
import requests
import PIL


# -------------------------------------------------------------------
# NOTEBOOK STYLING
# -------------------------------------------------------------------

class col:
    FG = '#E6F6FE'
    BG = '#FFFFFF'
    PRIMARY = '#BF1616'
    SECONDARY = '#615F5C'
    TERTIARY = '#F6F7F7'
    BLACK = '#000000'
    WHITE = '#FFFFFF'
    GRAY = '#AAB9C3'
    PINK = '#F68DA6'
    RED = '#CF2E2E'
    ORANGE = '#FF6801'
    YELLOW = '#FCB900'
    TURQUOISE = '#7BDCB5'
    GREEN = '#01D184'
    SKY_BLUE = '#8ED1FC'
    BLUE = '#0693E3'
    PURPLE = '#9B50E1'

def init_mpl(pyplot_module):
    cycle_colors = [
        col.RED, col.BLUE, col.GREEN,
        col.PURPLE, col.YELLOW, col.ORANGE,
        col.PINK, col.TURQUOISE, col.SKY_BLUE
    ]
    pyplot_module.rcParams['axes.prop_cycle'] = cycler(color=cycle_colors)
    #pyplot_module.rcParams['grid.color'] = col.PRIMARY

    
# -------------------------------------------------------------------
# MATPLOTLIB UTILS
# -------------------------------------------------------------------
    
def plot_waterfall(x, y, zs, ax):
    '''
    Plotting helper to create a 3D cascading waveform
    '''
    ax.axes.set_xlim(np.min(x), np.max(x))
    ax.axes.set_ylim(np.min(y), np.max(y))
    ax.axes.set_zlim(np.min(zs), np.max(zs))
    wf_2d_slices = []
    for yy in range(len(y)):
        wf_2d_slices.append(np.column_stack([x, zs[yy]]))
    ax.add_collection3d(
        PolyCollection(wf_2d_slices, facecolor=col.WHITE, edgecolor=col.BLUE),
        zs=y, zdir='y'
    )

# -------------------------------------------------------------------
# DATA LOADSING/MUNGING
# -------------------------------------------------------------------    

def urlimg(url):
    '''
    Pulls an image from the url and decodes it into an image
    object using pillow (because plt.imread is deprecated).
    '''
    img_data = BytesIO(requests.get(url).content)
    return PIL.Image.open(img_data)
    

# -------------------------------------------------------------------
# SCIPY/NUMPY HELPERS
# -------------------------------------------------------------------

def make_integrator_snapshots(solver, n_snapshots):
    '''
    Steps through an integrator solution and then slices
    the solution into snapshots
    '''
    _t = []
    _y = []
    while solver.status != 'finished':
        solver.step()
        _t.append(solver.t)
        _y.append(solver.y)

    _ss_step = int(len(_t)/int(n_snapshots))
    return  np.array(_t[::_ss_step]), np.array(_y[::_ss_step])

def inverse_svd(u, s, v, rank):
    '''
    Given an SVD, this will re-assemble the parts, taking only
    the first <rank> singular components.
    '''
    sv = np.matmul(np.diag(np.concatenate([s[:rank], np.zeros(len(s)-rank)])), v)
    return np.matmul(u, sv)