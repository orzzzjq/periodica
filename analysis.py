import numpy as np
import pandas as pd
from periodica.core import Periodica

import os

import time
os.getcwd()

file_path = 'tmp/data.initial_1_30Li'

with open(file_path, 'r') as f:
    data = f.readlines()

print(f"Loaded {len(data)} lines from {file_path}")

# Find the start index for the Atoms section
start_idx = 0
for i, line in enumerate(data):
    if line.startswith('Atoms'):
        # Skip the "Atoms" line and the empty line following it
        start_idx = i + 2
        break

# Extract the relevant data lines and split into columns
atom_data = [line.split() for line in data[start_idx:] if line.strip()]

# Create the dataframe using standard LAMMPS complete format columns
columns = ['id', 'type', 'charge', 'x', 'y', 'z', 'nx', 'ny', 'nz']
df = pd.DataFrame(atom_data, columns=columns)

# Ensure proper data types
df = df.astype({
    'id': int, 
    'type': int, 
    'charge': float, 
    'x': float, 
    'y': float, 
    'z': float, 
    'nx': int, 
    'ny': int, 
    'nz': int
})

# remove Li i.e. atom type 3
# unit cell only has atoms Si and S
df = df[df.type != 3]

# remove nx, ny, and nz
df = df[['id', 'type', 'charge', 'x', 'y', 'z']]

# Construct INPUT dictionary for Periodica
# The simulation box was 38.440 in all directions based on the data block
L = 38.440
U = np.eye(3) * L

# Randomly select 100 points
df = df.sample(n=500, random_state=42)

points = df[['x', 'y', 'z']].values.T

# Add a small amount of random noise to coordinates to break any perfectly symmetric 
# geometric degeneracies that can cause the exact-arithmetic C++ underlying library 
# to freeze or compute infinitely.
np.random.seed(42)
#noise = np.random.uniform(-1e-5, 1e-5, points.shape)
#points = points #+ noise

INPUT = {
    "d": 3,
    "U": U,
    "n_points": points.shape[1],
    "points": points
}


# Run periodica 
# Model the void space using the voronoi filtration
print("Quotient complex type: voronoi ")
start_time = time.time()

p = Periodica()
p.set_geometry(INPUT)
print("geometry set...")
p.quotient_complex('voronoi')
print("constructing merge tree...")
p.merge_tree()
# p.print_merge_tree()
# p.plot_all_descriptors(show=True, same_range=False)

end_time = time.time()
print(f"\nTime taken: {end_time - start_time:.4f} seconds")



#p.plot_all_descriptors(show=False, same_range=False)
#if show_geometry:
#    if TYPE != 'delaunay':
#        p.plot_geometry(TYPE, show=False, slidebar=True, use_circumcenter=False)
#        p.plot_geometry(TYPE, show=True, slidebar=True,use_circumcenter=False )

