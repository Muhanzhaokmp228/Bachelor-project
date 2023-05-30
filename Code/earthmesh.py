# -*- coding: utf-8 -*-
"""
@Project: BSC Youran and Muhan's project
@File: earthmesh.py

@Description: try to get it right this time
What I want to do: start with a good icosphere, and then cut vertices that are "on land"


"""

import numpy as np
from icosphere import icosphere
from scipy.interpolate import NearestNDInterpolator
from matplotlib.pyplot import imread
from gridmesh import dump_obj_no_uv_

VIEWER = "mayavi"
if VIEWER == "mayavi":
    try:
        from mayavi import mlab
    except ImportError:
        VIEWER = "plotly"
if VIEWER == "plotly":
    try:
        import plotly.graph_objects as go
    except ImportError:
        VIEWER = "matplotlib"
if VIEWER == "matplotlib":
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
    

__author__ = "François Lauze, University of Copenhagen"
__date__ = "5/29/23"
__version__ = "0.0.1"


# contains a segmented map of the world
LAND_FILE = "mymap.png"



def affine_map1D(xmin, xmax, ymin, ymax):
    """
    Affine map from [xmin, xmax] to [ymin, ymax]
    """
    return lambda x: ymin + (ymax - ymin) * (x - xmin) / (xmax - xmin)




class OceanMesh:
    """
    A class that represents a mesh of the earth ocean surface, just easier for encapsulation...
    """
    def __init__(self, resolution=5):
        """
        Constructor
        """
        self.resolution = resolution
        self.vertices = None
        self.faces = None
        self.land_map = imread(LAND_FILE)[..., 0]
        self.wi, self.wj = self.land_map.shape
        
        # create the icosphere
        self.vertices, self.faces = icosphere(self.resolution)
        
        # ocean nearest neighbor interpolator
        # interp(i, j) = 1 if (i, j) is in the ocean, 0 otherwise
        xi, yi = np.mgrid[0:self.wi, 0:self.wj]
        self.interp = NearestNDInterpolator(list(zip(xi.flatten(), yi.flatten())), self.land_map.astype(float).flatten())
        self.theta2i = affine_map1D(0, np.pi, 0, self.wi - 1)
        # 180 deg west is column 0, 180 deg east is column wj - 1
        self.phi2j = affine_map1D(-np.pi, np.pi, 0, self.wj - 1)#, 0)
        
    
    def mesh2map(self, x, y, z):
        """
        Map a point on the mesh to a point on the map
        """
        # convert to spherical coordinates
        theta = np.arccos(z)
        phi = np.arctan2(y, x)
        # convert to map coordinates thanks to the map coordinates computed in __init__()
        i = self.theta2i(theta)
        j = self.phi2j(phi)
        return i, j
    
    def filter_mesh(self):
        """
        Filter the mesh to remove vertices that are on land
        """
        nan = np.nan
        
        # convert vertices to map coordinates
        i, j = self.mesh2map(*self.vertices.T)
        # get the land map value at each vertex
        land_map_values = self.interp(i, j)
        # keep only the vertices that are in the ocean
        keep = land_map_values == 1
        # a bit of cleanup: I need to cut vertices and faces and to renumber them
        vertices = self.vertices.copy()
        # set z-coordinate to nan for vertices that are on land
        vertices[~keep, 2] = nan
        idx_v = np.where(keep)[0]
        
        # valid faces are those that have 3 valid vertices, i.e. no z coordinate is nan
        valid_face_1 = np.where(np.isfinite(vertices[self.faces[:, 0], 2]))[0]
        valid_face_2 = np.where(np.isfinite(vertices[self.faces[:, 1], 2]))[0]
        valid_face_3 = np.where(np.isfinite(vertices[self.faces[:, 2], 2]))[0]
        valid_faces = np.intersect1d(np.intersect1d(valid_face_1, valid_face_2), valid_face_3)
        
        # renumber the faces via the "remapper"
        dict_mapper = {idx_v[i]: i for i in range(len(idx_v))}
        remapper = np.vectorize(lambda x: dict_mapper[x])
        faces = remapper(self.faces[valid_faces])
        vertices = vertices[idx_v]
        # hopefully done!
        return vertices, faces
    
        


def view_mesh_plotly(vertices, faces):
    """
    View the mesh using plotly
    """
    # create the figure
    fig = go.Figure(data=[go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2])])
    # show the figure
    fig.show()
    
def view_mesh_mayavi(vertices, faces):
    """
    View the mesh using mayavi
    """
    mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces)
    mlab.show()
    
def view_mesh_matplotlib(vertices, faces):
    """
    View the mesh using matplotlib
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces)
    plt.show()
    
def view_mesh(vertices, faces):
    """
    View the mesh
    """
    if VIEWER == "plotly":
        view_mesh_plotly(vertices, faces)
    elif VIEWER == "mayavi":
        view_mesh_mayavi(vertices, faces)
    elif VIEWER == "matplotlib":
        view_mesh_matplotlib(vertices, faces)
    else:
        raise ValueError(f"Unknown viewer {VIEWER}")
    
    
if __name__ == "__main__":
    # create the mesh
    resolution = 2000
    mesh = OceanMesh(resolution=resolution)
    # filter the mesh
    vertices, faces = mesh.filter_mesh()
    # plot the mesh
    view_mesh(vertices, faces)
    # dump the mesh
    dump_obj_no_uv_(f"ocean_mesh_{resolution:03}.obj", vertices.T, faces)
    