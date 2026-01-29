import numpy as np

import matplotlib.pyplot as plt

# Replace with your actual file path
filename = '/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/data/20240425_PS-beads_60x_1mm_42mW_cw-853_grating3_glass_1.txt'

# Load data (assumes comma-delimited text file)
data = np.loadtxt(filename, delimiter=',')
# Convert wavelength (in nm) to wavenumber (in cm^-1)
# Wavenumber (cm^-1) = 1e7 / wavelength (nm)
# Calculate wavenumber for 785 nm excitation
excitation_wavenumber = 1e7 / 785  # in cm^-1
x = excitation_wavenumber - (1e7 / data[278:1024, 0])  # Raman shift in cm^-1

y = data[278:1024, 1]

fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
fig.patch.set_alpha(0.0)  # Transparent background for the figure
ax.set_facecolor('none')  # Transparent background for the axes

ax.plot(x, y, linestyle='-', color=plt.get_cmap('tab20')(5), marker=None, linewidth=2)
ax.axvline(x=1006, color='white', linestyle='--', linewidth=1.5)
ax.set_xlabel('Raman Shift (cm$^{-1}$)', color='white', fontsize=24)
ax.set_ylabel('Intensity (a.u.)', color='white', fontsize=24)

# Set tick parameters to white and increase font size
ax.tick_params(axis='x', colors='white', labelsize=20)
ax.tick_params(axis='y', colors='white', labelsize=20)

# Set left and bottom spines (borders) to white, hide top and right
ax.spines['left'].set_edgecolor('white')
ax.spines['bottom'].set_edgecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(True, color='white', alpha=0.2)
plt.tight_layout()
plt.savefig('polystyrene_beads_spectrum.png', dpi=300, transparent=True, bbox_inches='tight')
