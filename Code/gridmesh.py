"""
@Project: PS_2023 from RotInvFeatures
@filename: gridmesh.py

@description: regular grid meshes for single view PS

@Author François (This is some modified version from own code at Uni )


To do: some refactor? especially the grid_mesh_cut() function
      which removes v that fall outside a mask ans well
      as adjacent f.
"""

import os
import numpy as np
from typing import Union, Tuple
import matplotlib.pyplot as plt

__version__ = "0.0.1"
__author__ = "François Lauze"


def dump_obj_no_uv_(filename, vertex_array: np.ndarray, faces_array: Union[np.ndarray, list]) -> None:
    """
    Dump the grid mesh in an OBJ file, no UV content.

    :param filename: name of OBJ file
    :param vertex_array: a vertex array
    :param faces_array: list or array of f as triples of vertex indices
    :return: None.
    """
    lv = vertex_array.shape[1]
    obj = open(filename, 'w')
    obj.write('\n')
    for i in range(lv):
        obj.write(f'v {vertex_array[0, i]} {vertex_array[1, i]} {vertex_array[2, i]}\n')
    obj.write('\n')
    for v1, v2, v3 in faces_array:
        obj.write(f'f {v1 + 1} {v2 + 1} {v3 + 1}\n')
    obj.write('\n')
    obj.close()



class GridMesh:
    """
    Create a simple grid mesh, i.e. a mesh on a regular grid.

    GridMesh(m, n) creates a (m,n,1) grid, m vertical x n horizontal
    vertex points, and by default, their extent is [-1,1]^2x{0}
    (i.e. z-coordinates are 0). This can be changed  by
    - by changing vertical extent YExtent or horizontal extent XExtent
    - or specifying a  3D affine transformation which will be applied
      to default vertex coordinates.
    Both changes can be applied simultaneously for convenience.

    The class can compute UV coordinates (for texturing) and save the mesh
    with or without texturing as an OBJ.
    """
    XExtent = (-1.0, 1.0)
    YExtent = (-1.0, 1.0)
    DefaultTransform = (np.eye(3), np.zeros(3))

    def __init__(self, m: int, n: int) -> None:
        """
        :param m: number of vertical points
        :param n: number of horizontal points
        """
        self.m = m
        self.n = n
        self.vertices, self.faces, self.idx = self.create_grid()
        self.lv = len(self.vertices)
        self.transform = GridMesh.DefaultTransform
        self.XExtent = GridMesh.XExtent
        self.YExtent = GridMesh.YExtent

    def __repr__(self) -> str:
        return f"GridMesh(y-extent: {self.m}, x-extent: {self.n})"

    def __str__(self) -> str:
        return repr(self)

    def create_grid(self) -> tuple:
        """
        Create a grid with v and triangles.

        :return: list of vertex coordinate pairs and face list.
        """
        y, x = np.mgrid[0:self.m, 0:self.n]
        # list of v
        vertices = list(zip(y.flatten(), x.flatten()))
        # dictionary vertex_pair : index
        idx = dict(zip(vertices, range(len(vertices))))

        faces = []
        for j in range(self.n - 1):
            for i in range(self.m - 1):
                faces.append((idx[(i, j)], idx[(i, j + 1)], idx[(i + 1, j)]))
                faces.append((idx[(i, j + 1)], idx[(i + 1, j + 1)], idx[(i + 1, j)]))

        return vertices, faces, idx

    def set_extents(self, tx: tuple = XExtent, ty: tuple = YExtent) -> None:
        """
        Set new x and y extents for the grid.

        By default, reset the extents to [-1, 1]
        :param tx: extent interval (x_min, x_max)
        :param ty: extent interval (y_min, y_max)
        """
        self.XExtent = tx
        self.YExtent = ty

    def set_grid_extents(self) -> None:
        """
        Set the extents to the grid size.

        self.XExtent is set to (0, n-1) and self.YExtent is set to (0, m-1)
        """
        self.YExtent = (0, self.m - 1)
        self.XExtent = (0, self.n - 1)

    def set_transform(self, T: tuple) -> None:
        """
        Set an affine transformation used to produce 3D points from the grid.

        :param T: pair (albedo, t) affine transformation where albedo is a 3x3 matrix and t
                is a 3-translation
        :type T: tuple | list
        :return: None.
        """
        self.transform = T

    def vertices3D(self) -> np.ndarray:
        """
        Create a 3D vertex list from the list of vertex coords and affine transformation.

        :return: a (3, self.m * self.n) numpy array containing all the vertex coordinates, each column
                 contains the (x, y, z) coordinates
        """
        x = np.linspace(*self.XExtent, num=self.n, endpoint=True)
        y = np.linspace(*self.YExtent, num=self.m, endpoint=True)

        vertex_array = np.zeros((3, self.lv))
        for k, (i, j) in enumerate(self.vertices):
            vertex_array[0, k] = x[j]
            vertex_array[1, k] = y[i]

        A, t = self.transform
        vertex_array = A @ vertex_array
        t.shape = (3, 1)
        vertex_array += t
        return vertex_array

    def deproject(self, K: np.ndarray, z: np.ndarray,
                  uv: Union[tuple,None] = None) -> np.ndarray:
        """
        Create a 3D vertex list by deprojecting it from a perspective pinhole camera.

        :param K: 3x3 camera's intrinsic parameters' matrix
        :param z: (self.m, self.n) depth map
        :param uv: tuple of image coordinates (u,v) as would be produced by np.mgrid[]
                or None, in which case they are obtained from self.v (i.e. from the grid),

        :return: a (3, self.m * self.n) numpy array containing the 3D coordinates each column
                 contains the (x, y, z) coordinates
        """
        if uv is None:
            vertex_array = np.array(self.vertices)
            vertex_array[:, [0, 1]] = vertex_array[:, [1, 0]]
        else:
            v, u = np.mgrid[0:self.m, 0:self.n]
            vertex_array = np.vstack((u, v))
        vertex_array = np.vstack((vertex_array.T, np.ones(self.lv)))
        vertex_array = np.linalg.inv(K) @ vertex_array
        vertex_array *= np.reshape(z, (1, -1))
        return vertex_array

    def uv_coords(self) -> np.ndarray:
        """
        Create uv-coordinate map for the grid mesh.
        
        :return: a numpy array containing the 2D UV points coordinates
        """
        u = np.linspace(0, 1, num=self.m, endpoint=True)
        v = np.linspace(0, 1, num=self.n, endpoint=True)
        UV = np.zeros((2, self.lv))
        for k, (i, j) in enumerate(self.vertices):
            UV[0, k] = u[i]
            UV[1, k] = v[j]
        return UV

    def get_mesh_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return v, UV and f for the grid mesh.
        
        :return: triple of v, uv-coordinates and f
        """
        vertex_array = self.vertices3D()
        UV = self.uv_coords()
        return vertex_array, UV, np.array(self.faces)

    def dump_obj_no_uv(self, filename, varray: Union[np.ndarray, None] = None) -> None:
        """
        Dump the grid mesh in an OBJ file, no UV content.

        :param filename: name of OBJ file
        :param varray: if not None, a vertex array that should be compatible with the grid (TO TEST?).
        :return: None.
        """
        vertex_array = self.vertices3D() if varray is None else varray
        dump_obj_no_uv_(filename, vertex_array, self.faces)

    def dump_obj_uv(self, filename, varray: Union[np.ndarray, None] = None, texture: str = "") -> None:
        """
        Dump the grid mesh in an OBJ file, no UV content.

        :param filename: name of OBJ file
        :type filename: str
        :param varray: if not None, a vertex array that should be compatible with the grid (TO TEST?).
        :param texture: filename of a 'texture file'; i.e., an image
        :type texture: str
        :return: None
        """
        vertex_array = self.vertices3D() if varray is None else varray
        UV = self.uv_coords()
        obj = open(filename, 'w')
        obj.write('\n')

        if texture != "":
            # create associated mtl file
            # ambient and specular components set to 0
            mtlname = os.path.splitext(filename)[0] + '.mtl'
            with open(mtlname, 'w') as mtl:
                mtl.write(f'newmtl material_1\nmap_Kd {texture}\n')
                mtl.write('Ka 0.000 0.000 0.000 # black\n')
                mtl.write('Kd 1.000 1.000 1.000 # white\n')
                mtl.write('Ks 0.000 0.000 0.000 # black\n')
                mtl.write('Ns 0.0\n')
            obj.write(f'mtllib {mtlname}\n')
            obj.write('\n')

        for i in range(self.lv):
            obj.write(f'v {vertex_array[0, i]} {vertex_array[1, i]} {vertex_array[2, i]}\n')
        obj.write('\n')
        for i in range(self.lv):
            obj.write(f'vt {UV[0, i]} {UV[1, i]}\n')
        if texture != "":
            obj.write('usemtl material_1\n')
        for (v1, v2, v3) in self.faces:
            obj.write(f'f {v1 + 1}/{v1 + 1} {v2 + 1}/{v2 + 1} {v3 + 1}/{v3 + 1}\n')
        obj.write('\n')
        obj.close()

# To do?  refactor/move to somewhere else?
def grid_mesh_cut(vertices: np.ndarray, faces: Union[list, np.ndarray], mask: np.ndarray) -> tuple:
    """
    Remove v and corresponding f from a grid mesh according to a mask.
    
    intensities need to assume that the flattened mask has pixels ordered in the same way
    as the mesh v.
    
    :param vertices: vertex array, size (3, N)
    :param faces:  list of f
    :param mask: binary mask array, size (m, n) with N = m * n
    :return: tuple (cut_vertices, cut_faces)
    """
    cut_vertices = vertices.copy()
    cut_faces = np.array(faces)
    cut_vertex_indices = (mask > 0).flatten()
    
    # mark v outside the mask as invalid
    cut_vertices[2, ~cut_vertex_indices] = float('nan')
    
    # check for f which contain an invalid vertex
    i1, i2, i3 = zip(*cut_faces)
    i1_good = np.isfinite(cut_vertices[2, i1])
    i2_good = np.isfinite(cut_vertices[2, i2])
    i3_good = np.isfinite(cut_vertices[2, i3])
    good_face_indices = np.logical_and(i1_good, np.logical_and(i2_good, i3_good))
    
    # cut vertex list
    cut_vertices = vertices[:, cut_vertex_indices]

    # compute vertex renumbering function from a dictionary f(old_index) = new_index
    keys = np.where(cut_vertex_indices)[0]
    items = range(len(keys))
    dict_f = dict(zip(keys, items))
    f = np.vectorize(lambda x : dict_f[x])

    # cut face list and renumber indices
    cut_faces = cut_faces[good_face_indices]
    cut_faces = f(cut_faces)
    
    return cut_vertices, cut_faces




if __name__ == "__main__":
    m, n = 31, 35
    gm = GridMesh(m, n)
    print(gm.vertices)

    y, x = np.mgrid[-1:1:m*1j, -1:1:n*1j]
    z = 1 + 10.0 * np.exp(-(x ** 2 + y ** 2)/0.0625)

    K = np.array([[3., 0.0, n/2],
                 [0.0, 3., m/2],
                 [0.0, 0.0, 1.0]])
    v = gm.deproject(K, z)
    print(v)
    gm.dump_obj_no_uv('foo_notexture.obj', varray=v)
    
    mask = (x **2 + y **2 ) < 0.81
    vertices, _, faces = gm.get_mesh_components()
    cverts, cfaces = grid_mesh_cut(v, faces, mask)
    dump_obj_no_uv_('cut_foo_notexture.obj', cverts, cfaces)
    
    