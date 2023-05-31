% mesh each mitochondrion extracted from Chenhao's segemnted scan
% save the mesh in a mat file that also contains
% - the extracted volume
% - the cristae_junction list (possibly empty)
% - the mitochondrion number, from cc3d.connected_component() label value
%   this should also be encoded in the file name
% - the min coordinates (minx, miny, minz) of the mitochondrion in the 
%   original volume mito_volume.npy. Beware that each extracted volume was 
%   padded by zeros so one should have, if nx, ny, nz = mito.shape, 
%   mito_volume[minx:minx+nx-2, miny:miny+ny-2, minz:minz+nz-2] =
%   mito[1:-1, 1:-1, 1:-1]
% beware the +/-1 offset between Python and Matlab!


mitolist = dir('mito_data*.mat')

for i = 1 : length(mitolist)
    % parse the name
    fname = mitolist(i).name;
    mito_number = str2num(fname(10:12));
    fprintf("Now processing file %s\n", fname);
    load(fname);
    [vertices, faces] = isosurface(mito);
    % saving...
    min_coords = min_coors;
    save(fname, "mito_number", "mito", "cristae_junction", "min_coords", "faces", "vertices");
end
