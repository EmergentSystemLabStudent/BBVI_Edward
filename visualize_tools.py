import numpy as np

def circle(a,b,r):
    theta = np.linspace(-1 * np.pi, 1 * np.pi, 100)
    x = r * np.cos(theta)+a
    y = r * np.sin(theta)+b
    return x,y

def ellipse(a,b,alpha,center_x=0.0,center_y=0.0):
    theta = np.linspace(-1 * np.pi, 1 * np.pi, 100)
    x = a * np.sin(theta)
    y = b * np.cos(theta)
    c=np.cos(alpha)
    s=np.sin(alpha)
    x2=x*c-y*s+center_x
    y2=x*s+y*c+center_y
    return x2,y2

def covariance_ellipse(covariance):
    la, v = np.linalg.eig(covariance)
    angle=np.arctan2(v[1,0],v[0,0])
    std = np.sqrt(la)
    return std[0],std[1],angle

def change_aspect_ratio(ax, ratio):
    '''
    This function change aspect ratio of figure.
    Parameters:
        ax: ax (matplotlit.pyplot.subplots())
            Axes object
        ratio: float or int
            relative x axis width compared to y axis width.
    '''
    aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(aspect)