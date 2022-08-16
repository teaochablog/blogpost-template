import os
import re
import numpy as np
import matplotlib.pyplot as plt
import requests
import PIL
import seaborn as sns
import pandas as pd
from matplotlib.collections import PolyCollection
from io import BytesIO
from IPython.core.display import HTML
from IPython.core.pylabtools import print_figure
from base64 import b64encode

plt.rcParams['figure.dpi'] = 300

# -------------------------------------------------------------------
# NOTEBOOK STYLING
# -------------------------------------------------------------------

class col:
    FG = '#E6F6FE'
    BG = '#FFFFFF'
    PRIMARY = '#BF1616'
    SECONDARY = '#615F5C'
    TERTIARY = '#F6F7F7'
    NEUTRAL_LIGHTER = '#F5F5F5'
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

def init_theme(_sns):
    _sns.set_theme(style='darkgrid')
    _sns.set_style('darkgrid', {'axes.facecolor': col.NEUTRAL_LIGHTER })
    _sns.set_palette("Paired")  

    
# -------------------------------------------------------------------
# MATPLOTLIB UTILS
# -------------------------------------------------------------------

def html_df(df, fignum, figcaption):
    '''Renders as pandas dataframe as a figure with a caption.
    '''
    df_html = df.to_html()
    html = '''
        <figure
            class="nb-generated-diagram"
            align="center"
            style="display:flex; align-items:center; flex-flow:column;"
        >
            {}
            <figcaption>Figure {}: {}</figcaption>
        </figure>
    '''.format(df_html, fignum, figcaption)
    return HTML(html)

def html_fig(fig, fignum, figcaption, source='TeaochaDesign'):
    '''Takes a matplotlib figure and turns it into an HTML
    figure instead.
    '''
    fig_b64 = b64encode(print_figure(fig)).decode("utf-8")
    img_data = f'data:image/png;base64,{fig_b64}'
    if source == '' or source == None:
        source = ''
    else:
        source = f' (Source: {source})'
    html = '''
        <figure class="nb-generated-diagram" align="center">
            <img src="{}">
            <figcaption>Figure {}: {}{}</figcaption>
        </figure>
    '''.format(img_data, fignum, figcaption, source)
    plt.close();
    return HTML(html)

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

def image_bootstrapper(images_dir: str, x_crop: float=0.5, y_crop: float=0.5):
    '''Given a directory containing images, returns a generator for
    subsamples of the images in that directory.
    
    Params:
        images_dir:
            The directory containing the images
        x_crop:
            The ratio of the image to randomly crop in the x axis
        y_crop:
            The ratio of the image to randomly crop in the y axis
    '''
    image_paths = [
        img_path for img_path in os.listdir(images_dir)
        if re.search(r'\.(png|jpg|jpeg|bmp)$', img_path)
    ]
    images = [
        PIL.Image.open(os.path.join(images_dir, image_path))
        for image_path in image_paths
    ]
    
    while True:
        img = images[np.random.randint(len(images))]
        crop_width = np.math.floor(img.width * x_crop)
        crop_max_x = img.width - crop_width
        crop_x = np.random.randint(crop_max_x) if crop_max_x > 0 else 0
        crop_height = np.math.floor(img.height * y_crop)
        crop_max_y = img.height - crop_height
        crop_y = np.random.randint(crop_max_y) if crop_max_y > 0 else 0
        
        cropped = img.crop((
            crop_x,
            crop_y,
            crop_x + crop_width,
            crop_y + crop_height
        ))
        yield cropped

def kl(P,Q):
    """KL-Divergence of Q with respect to P.
    """
    # Epsilon is used here to avoid conditional code for
    # checking that neither P nor Q is equal to 0.
    epsilon = 0.00001
    P = P + epsilon
    Q = Q + epsilon
    
    divergence = np.sum(P*np.log(P/Q))
    return divergence