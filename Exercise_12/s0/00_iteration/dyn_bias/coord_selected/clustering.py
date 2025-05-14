import numpy as np
import random 


def read_xyz(filename):
    """Read XYZ file and return atom names and coordinates

    Args:
        filename:  Name of xyz data file

    Returns:
        atom_names: Element symbols of all the atoms
        coords: Cartesian coordinates for every frame.
    """
    coors = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                natm = int(line)  # Read number of atoms
                next(f)     # Skip over comments
                atom_names = []
                xyz = []
                for i in range(natm):
                    line = next(f).split()
                    atom_names.append(line[0])
                    xyz.append(
                        [float(line[1]), float(line[2]), float(line[3])])
                coors.append(xyz)
            except (TypeError, IOError, IndexError, StopIteration):
                raise ValueError('Incorrect XYZ file format')

    return atom_names, coors


def write_xyz(filename, atoms, coords): 
    """Write atom names and coordinate data to XYZ file

    Args:
        filename:   Name of xyz data file
        atoms:      Iterable of atom names
        coords:     Coordinates, must be of shape nimages*natoms*3
    """ 
    natoms = len(atoms)
    with open(filename, 'w') as f:
        for i, X in enumerate(np.atleast_3d(coords)):
            f.write("%d\n" % natoms)
            f.write("Frame %d\n" % i)
            for a, Xa in zip(atoms, X): 
                f.write(" {:3} {:21.12f} {:21.12f} {:21.12f}\n".format(a, *Xa)) 


# -----------------------------------------------------------------


cv = []
time = []

with open('../b120/COLVAR','r') as file:
     for line in file:
         if '#' not in line:
             time.append(float(line.split()[0]))
             cv.append(float(line.split()[3]))


cv = np.array(cv)
time = np.array(time)

print ('Total number of data', len(cv))


bin_min = -3.5 
bin_max = 4.0

Nbin = 50  
ris = (bin_max-bin_min)/Nbin

cluster = np.zeros(Nbin) 

cluster_label = {}

print('Min, Max, Nbin, ris: ', bin_min, bin_max, Nbin, ris)

x_bin = []
for i in range(Nbin):
    cluster_label[i] = []
    for j in range(len(cv)):
        lower = (bin_min)+(i*ris)
        upper = (bin_min+ris)+(i*ris)
        if j == 0 :
           x_bin.append([lower,upper])
        if cv[j] >= lower and cv[j] < upper:
           cluster[i] += 1
           cluster_label[i].append(j)    


print ('Clustering:')
for i in range(Nbin):
    print (x_bin[i][0],x_bin[i][1],cluster[i])

print('Total data clustered', sum(cluster))


Nsample = 20 
#print (cluster_label[0])
selected_frame = []
sel_count = 0
for i in range(Nbin):
#    print(cluster[i]) 
    if cluster[i] < Nsample:
       selected_frame.append(random.sample(cluster_label[i], int(cluster[i])))
       print(i,' Bin frame selected:',cluster[i])
       sel_count += cluster[i]
    else: 
       selected_frame.append(random.sample(cluster_label[i], Nsample))
       print(i,' Bin frame selected:',Nsample)
       sel_count += Nsample 

print('Selected frames=',sel_count)

selected_frame = [item for sublist in selected_frame for item in sublist]
#print(selected_frame)

for i in range(len(cv)):
    if i in selected_frame:
       print (i,cv[i],time[i])

print('PRINT BIAS')

#filtering bias and double check cv 
with open('../b120/COLVAR','r') as file:
     for i,line in enumerate(file):
         if i in selected_frame:
            print(line,  end='')

print ('TRAJ')

atoms, coords = read_xyz('../b120/dump.xyz')

print(len(coords))
selected_coords = [coords[i] for i in sorted(selected_frame)]

print('Selected coords', len(selected_coords))

write_xyz('coords_selected.xyz', atoms, selected_coords)
