import numpy as np
import math
import igl

# distance between two points on a sphere 
# https://www.cuemath.com/trigonometry/trigonometric-table/
# Source: https://www.math.ksu.edu/~dbski/writings/haversine.pdf

def distance(r,vec1,vec2):
    # Arc Distance
    dist = r * np.arccos(np.dot(vec1,vec2)/r**2)
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

# Calculating the distance between all pairs of points on the mesh
def pair_distance_mesh(vecs, faces, samples):
    npts = np.shape(samples)[0]
    dist = []
    point_mesh = igl.point_mesh_squared_distance(samples, vecs, faces)
    sqrD, face_idx, cvecs = point_mesh
    for i in range(npts):
        for j in range(i+1, npts):
            # reshape to (1,3) array
            svec = cvecs[i]
            tvec = cvecs[j]
            d = igl.exact_geodesic(vecs, faces, svec, tvec) # distance between the closest vertices
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

def samples_uniform_sphere(n):
    # Generating z coordinates with radius = 1
    z = np.random.uniform(-1,1,n)

    # Generating azimuthal angles
    phi = np.random.uniform(0,2*math.pi,n)

    # Generating x and y coordinates
    x = np.sqrt(1-z**2)*np.cos(phi)
    y = np.sqrt(1-z**2)*np.sin(phi)
    samples = np.array([x,y,z]).T
    return samples

def sample_faces(vertices, faces, num_samples):
    # calculate the area of each triangle
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    areas = np.linalg.norm(np.cross(b-a, c-a), axis=1) / 2

    # normalize areas to get probability distribution
    probabilities = areas / np.sum(areas)

    # randomly choose triangles to sample from
    face_indices = np.random.choice(len(faces), size=num_samples, p=probabilities)

    # generate random points within each selected triangle
    u = np.random.rand(num_samples, 1)
    v = np.random.rand(num_samples, 1)
    a = vertices[faces[face_indices, 0]]
    b = vertices[faces[face_indices, 1]]
    c = vertices[faces[face_indices, 2]]
    points = (1 - u) * a + (u * (1 - v)) * b + (u * v) * c

    return points