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
    u = vecs[1] - vecs[0]
    v = vecs[2] - vecs[0]
    n = np.cross(u, v)
    w = point - vecs[0]

    gamma = np.dot(np.cross(u, w), n) / np.dot(n, n)
    beta = np.dot(np.cross(w, v), n) / np.dot(n, n)
    alpha = 1 - gamma - beta

    y = alpha * vecs[0] + beta * vecs[1] + gamma * vecs[2]
    return y
    # if alpha >= 0 and alpha <= 1 and beta >= 0 and beta <= 1 and gamma >= 0 and gamma <= 1:

def proj_points_to_mesh(vecs, faces, samples):
    npts = np.shape(samples)[0]
    proj_points = []
    point_mesh = igl.point_mesh_squared_distance(samples, vecs, faces)
    for i in range(npts):
        # reshape to (1,3) array
        vs = samples[i].reshape(1,3)
        svec = faces[point_mesh[1][i]]
        y = project_point_to_mesh(vs, vecs[svec]).astype(float)
        proj_points.append(y)
    return proj_points

# Calculating the shortest path between two points on the mesh
def shortest_path(vecs, faces, vs, vt, svec ,tvec):

    shortest_path = math.inf

    # find the closest vertices to the source and target points
    svec_coords = vecs[svec] 
    tvec_coords = vecs[tvec]

    # find the shortest path between the source and target points
    dist_vs_vecs = np.linalg.norm(vs-svec_coords, axis=1) # distance between the source point and the closest vertices
    dist_vt_vecs = np.linalg.norm(vt-tvec_coords, axis=1) # distance between the target point and the closest vertices
    dist_vecs = igl.exact_geodesic(vecs, faces, svec, tvec) # distance between the closest vertices
    dist = dist_vs_vecs + dist_vecs + dist_vt_vecs # total distance
    shortest_path = np.min(dist)
    
    return shortest_path

# Calculating the distance between all pairs of points on the mesh
def pair_distance_mesh(vecs, faces, samples):
    npts = np.shape(samples)[0]
    dist = []
    samples = np.array(proj_points_to_mesh(vecs, faces, samples))
    point_mesh = igl.point_mesh_squared_distance(samples, vecs, faces)
    # point_mesh = igl.signed_distance(samples, vecs, faces)
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