import numpy as np
import math
import igl

# distance between two points on a sphere 
# https://www.cuemath.com/trigonometry/trigonometric-table/
# Source: https://www.math.ksu.edu/~dbski/writings/haversine.pdf

def distance(r,vec1,vec2):
    # pythagorean theorem
    d = np.linalg.norm(vec1-vec2)
    # Arc Distance
    dist = r * 2 * math.asin(d/2)
    return dist

def pairDistances(r, data):
    npts = np.shape(data)[0]
    dist = []
    for i in range(npts):
        for j in range(i+1, npts):
            dist.append(distance(r, data[i], data[j]))
    return dist

# Calculating the distance between all pairs of points
def surfaceArea(r):
    area = 4 * math.pi * r**2
    return area

# Ripley's K function for sphere
def ripleyK(r, data, radii):
    K = np.zeros_like(radii)
    area = surfaceArea(r)
    dist = pairDistances(r, data)
    intensity = len(dist) / area
    for i in range(len(radii)):
        K[i] = np.sum(dist < radii[i])
    K = K / intensity
    return K

#Projecting the point on the mesh
def project_point_to_mesh(point, vecs):
    v1 = vecs[1] - vecs[0]
    v2 = vecs[2] - vecs[0]
    
    # normal = np.cross(v1, v2) # Calculate the normal vector of the plane
    # # v = point - vecs[0] # Calculate direction vector of the point to the plane
    # vl = np.dot(point - vecs[0], normal) # Calculate the length of the projection vector
    # nl = np.dot(normal, normal) # Calculate the length of the normal vector
    # y = point - vl / nl * normal # Calculate the projection point
    A = vecs
    # Pmaxtrix = A * np.linalg.inv(A.T * A) * A.T
    # y = Pmaxtrix * point.T
    
    Pmatrix = np.dot(A, np.dot(np.linalg.inv(np.dot(A.T, A)), A.T))
    y = np.dot(Pmatrix, point.T)

    return y

# Calculating the shortest path between two points on the mesh
def shortest_path(vecs, faces, vs, vt, svec ,tvec):

    shortest_path = math.inf

    # # get the indices and coordinates of the closest vertices to the source and target points
    # ds = igl.point_mesh_squared_distance(vs, vecs, faces)
    # # ds = igl.signed_distance(vs, vecs, faces)
    # svec = faces[ds[1]]
    svec_coords = vecs[svec] # coordinate for the 3 closest vertices to the source
    
    # dt = igl.point_mesh_squared_distance(vt, vecs, faces)
    # #dt = igl.signed_distance(vt, vecs, faces)
    # tvec = faces[dt[1]]
    tvec_coords = vecs[tvec] # coordinate for the 3 closest vertices to the target

    # find the shortest path between the source and target points
    vs = project_point_to_mesh(vs, svec_coords).astype(float) # project the source point on the mesh
    dist_vs_vecs = np.linalg.norm(vs-svec_coords, axis=1) # distance between the source point and the closest vertices
    # print(dist_vs_vecs)
    vt = project_point_to_mesh(vt, tvec_coords).astype(float) # project the target point on the mesh
    dist_vt_vecs = np.linalg.norm(vt-tvec_coords, axis=1) # distance between the target point and the closest vertices
    # print(dist_vt_vecs)
    dist_vecs = igl.exact_geodesic(vecs, faces, svec, tvec) # distance between the closest vertices
    dist = dist_vs_vecs + dist_vecs + dist_vt_vecs # total distance
    shortest_path = np.min(dist)
    
    return shortest_path

# Calculating the distance between all pairs of points on the mesh
def pair_distance_mesh(vecs, faces, samples):
    npts = np.shape(samples)[0]
    dist = []
    point_mesh = igl.point_mesh_squared_distance(samples, vecs, faces)
    for i in range(npts):
        for j in range(i+1, npts):
            # reshape to (1,3) array
            vs = samples[i].reshape(1,3)
            vt = samples[j].reshape(1,3)
            svec = faces[point_mesh[1][i]]
            tvec = faces[point_mesh[1][j]]
            d = shortest_path(vecs, faces, vs, vt, svec, tvec)
            dist.append(d)
    return dist

# Calculating the surface area of the mesh
def mesh_area(vecs, faces):
    double_areas = igl.doublearea(vecs, faces)
    surface_area = np.sum(double_areas) / 2.0
    return surface_area

# Ripley's K function for mesh
def ripleyK_mesh(vecs, faces, data, radii):
    K = np.zeros_like(radii)
    area = mesh_area(vecs, faces)
    dists = pair_distance_mesh(vecs, faces, data)
    intensity = len(dists) / area
    for i in range(len(radii)):
        K[i] = np.sum(dists < radii[i])
    K = K / intensity
    return K