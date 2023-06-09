import numpy as np
import math
import igl
import ripleyK as rk
import ripleyKmito as rkm
from scipy.io import loadmat 
import os


def readmitodata(fname):
    """
    Expect a filename of the form "mito_data???.mat" 
    where ??? is the mito number (left padded by 0 is necessary).
    
    argument:
    fname -- string, filename of the mito data
    
    returns:
    mito_dict -- dictionary, keys are the names of the variables in the matlab file
    The keys are:
    'mito' -- the mitochondrion binary shaoe (np.ndarray, has been padded with zeros)
    'vertices' -- the vertices of the meshed mitochondrion (np.ndarray)  
    'faces' -- the faces of the meshed mitochondrion (np.ndarray)
    'cristae_junctions' -- the vertices of the cristae junctions (np.ndarray) or empty list if none
    'min_coords' -- the minimum coordinates of the mitochondrion (np.ndarray) in the original volume    
    'mito_number' -- the number of the mitochondrion (int) from the cc3d.component_labeling function,
                        this is the same as the number in the filename
    
    """
    mito_dict = loadmat(fname)
    # remove matlab meta data
    del mito_dict['__header__']
    del mito_dict['__version__']
    del mito_dict['__globals__']
    mito_dict['mito_number'] = int(mito_dict['mito_number'])
    # inverted results in matlab....
    mito_dict['vertices'], mito_dict['faces'] = mito_dict['faces'], mito_dict['vertices']
    # count starts at 0
    mito_dict['faces'] -= 1
    return mito_dict


# Calculating the distance between all pairs of points on the mesh
def pair_distance_mesh(vertices, faces, samples):
    npts = np.shape(samples)[0]
    dist = []
    sqrD, face_idx, cvecs = igl.point_mesh_squared_distance(samples, vertices, faces)
    # print(face_idx)
    # print("cvecs: ", cvecs)
    # print(faces[face_idx])
    # print(vecs[faces[face_idx]])
    for i in range(npts-1):
        # print(faces[face_idx])
        vs = np.array([faces[face_idx][i][1]])
        # print("vs:", vs)
        vt = np.array(faces[face_idx][i+1:,1:2])
        # print("vt:", vt)
        d = igl.exact_geodesic(vertices, faces, vs, vt)
        # print("d: ", d)
        dist.append(d)
    return dist

# Calculating the surface area of the mesh
def mesh_area(vecs, faces):
    double_areas = igl.doublearea(vecs, faces)
    surface_area = np.sum(double_areas) / 2.0
    return surface_area

# Ripley's K function for mesh
def ripleyK_mesh(vecs, faces, data):
    area = mesh_area(vecs, faces)
    dists = pair_distance_mesh(vecs, faces, data)
    dists = np.hstack(dists)
    rmax = np.max(dists)
    radii = np.linspace(0, rmax, 50)
    K = np.zeros(len(radii))
    intensity = len(dists) / area
    for i in range(len(radii)):
        K[i] = np.sum(dists < radii[i])
    K = K / intensity
    return radii, K

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "mito_data")
result_dir = os.path.join(current_dir, "mito_result")

for file_name in os.listdir(data_dir):
    if file_name.endswith(".mat"):
        file_path = os.path.join(data_dir, file_name)
        mito = readmitodata(file_path)
        vertices = np.array(mito['vertices'], dtype=np.float64)
        faces = np.array(mito['faces'], dtype=np.int32)
        cj = np.array(mito['cristae_junction'], dtype=np.float64)
        points = cj.T
        points = points[:, [1, 0, 2]].astype(np.float64)
        rmax = math.sqrt(rkm.mesh_area(vertices, faces) / 2)
        radii = np.linspace(0, rmax+rmax/8, 50)
        if cj.size:
            print(file_name)
            kt_mito = ripleyK_mesh(vertices, faces, points, radii)
            data = np.column_stack((radii, kt_mito))
            result_path = os.path.join(result_dir, os.path.splitext(file_name)[0] + ".csv")
            np.savetxt(result_path, data, delimiter=",", header='radii,kt_mito')
            print(file_name + " result saved")
        else:
            print("No cristae junctions found")