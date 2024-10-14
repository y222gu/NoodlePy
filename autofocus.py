import numpy as np
from ctypes import *
import time

from wasatch.WasatchBus    import WasatchBus
from wasatch.WasatchDevice import WasatchDevice

import matplotlib.pyplot as plt


def calculate_entropy(intensities):
    normalized_int = intensities / np.sum(intensities)
    entropy = -np.sum(normalized_int * np.log2(normalized_int))
    return entropy

def get_colors(num_colors):
    colors = plt.cm.jet(np.linspace(0, 1, num_colors))
    return colors

plt.ion()

import matplotlib.pyplot as plt

# Creating a figure with a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2)

# axes is a 2x2 numpy array of AxesSubplot objects
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

x1_data, y1_data = [], []
x2_data, y2_data = [], []

# Create the first subplot
ax1.set_xlabel('Z-axis position')
ax1.set_ylabel('Entropy')
ax1.set_title('Entropy minimization')
ax1.grid(True)

ax2.set_xlabel('Wavelength')
ax2.set_ylabel('Intensity')
ax2.set_title('Rough Focus')
ax2.grid(True)

ax3.set_xlabel('Z-axis position')
ax3.set_ylabel('Entropy')
ax3.set_title('Entropy minimization')
ax3.grid(True)

ax4.set_xlabel('Wavelength')
ax4.set_ylabel('Intensity')
ax4.set_title('Fine Focus')
ax4.grid(True)

plt.tight_layout()


def autofocus(exp_time):
    #Rough Autofocusing
    entropy_list = []
    start_time = time.time()
    #For polystyrene use 65 as upper limit
    #For biofluid, use 5 as upper limit
    ##TO EDIT: range of autofocus z-axis, (min, max, number of steps)
    z_spacing = 20
    z_axis_range = np.linspace(0, 100, z_spacing)
    #exp_time = 0.1
    # set to false if using the old fused quartz substrates, to true if using the new crystal quartz (large)
    quartzCrystal = False

    num_rep = 5
    data_point = 1024
  
    int_array = []

    
    for i in range(z_spacing):  # Replace 10 with the number of iterations you need
        z_pos = z_axis_range[i]
        # Choose axis to move and new position in micrometers
        axis = c_uint(3)
        pos = c_double(z_pos)
        # Move to a new position
        error = mcldll.MCL_SingleWriteN(pos, axis, handle)
        if error != 0:
            print("Error = ", error)
        # Wait for nanopositioner to settle
        time.sleep(0.025)
        # Read the new position
        position = mcldll.MCL_SingleReadN(axis, handle)
        intensities = []
        wavelengths = []

        colors = get_colors(z_spacing)
        device.change_setting("integration_time_ms", exp_time)
        for rep in range(num_rep):
            
            spectrum = device.hardware.get_line().data.spectrum

            for pixel in range(device.settings.pixels()):
                wavelengths.append(device.settings.wavelengths[pixel])
                intensities.append(spectrum[pixel])
                intensity = spectrum[pixel]
                # or: spectrum = device.acquire_data().spectrum
                # if quartzCrystal:
                #    intensity = intensity[pixelsToCrop:]
            
            int_array.append(intensities)
        
        wavelengths = np.array(wavelengths).reshape((num_rep, data_point))
        intensities = np.array(intensities).reshape((num_rep, data_point))


        ent = calculate_entropy(np.median(intensities,0))
        entropy_list.append(ent)

        x1_data.append(position)
        y1_data.append(ent) 

        # Clear previous plots
        ax1.clear()
        #ax2.clear()

        # Plot updated data
        ax1.plot(x1_data, y1_data, label='Entropy')
        ax1.scatter([x1_data[-1]], [y1_data[-1]], color='red')  # Highlight the latest point

        ax2.plot(wavelengths[0], np.mean(intensities,0), color=colors[i], label=f'Line {i+1}')

        # ax2.plot(intensities)
        plt.draw()
        plt.pause(0.1)

    plt.show()
    print("Time elapsed = ", time.time()-start_time)
    entropy_graph = entropy_list
    z_axis = z_axis_range
   
    k = np.argmin(entropy_list)
    fineMin = z_axis_range[np.clip(k-1,0,len(z_axis_range)-1)]
    fineMax = z_axis_range[np.clip(k+1,0,len(z_axis_range)-1)]
    print("The fine range is: ", fineMin, fineMax)

    #Fine Autofocusing
    entropy_list = []
    z_axis_range = np.linspace(fineMin,fineMax,10)

    num_rep = 5
    data_point = 1024
    # if quartzCrystal:
    #     data_point -= pixelsToCrop
    int_array = []
    
    for z_pos in z_axis_range:
        # Choose axis to move and new position in micrometers
        axis = c_uint(3)
        pos = c_double(z_pos)
        # Move to a new position
        error = mcldll.MCL_SingleWriteN(pos, axis, handle)
        if error != 0:
            print("Error = ", error)

        # Wait for nanopositioner to settle
        time.sleep(0.025)
        # Read the new position
        position = mcldll.MCL_SingleReadN(axis, handle)
        
        intensities = []
        wavelengths = []

        colors = get_colors(z_spacing)

        for rep in range(num_rep):
            
            spectrum = device.hardware.get_line().data.spectrum

            for pixel in range(device.settings.pixels()):
                wavelengths.append(device.settings.wavelengths[pixel])
                intensities.append(spectrum[pixel])
                intensity = spectrum[pixel]
                # or: spectrum = device.acquire_data().spectrum
                # if quartzCrystal:
                #    intensity = intensity[pixelsToCrop:]
            
            int_array.append(intensities)
        
        wavelengths = np.array(wavelengths).reshape((num_rep, data_point))
        intensities = np.array(intensities).reshape((num_rep, data_point))


        ent = calculate_entropy(np.median(intensities,0))
        entropy_list.append(ent)

        x2_data.append(position)
        y2_data.append(ent) 

        # Clear previous plots
        ax3.clear()
        #ax2.clear()

        # Plot updated data
        ax3.plot(x2_data, y2_data, label='Entropy')
        ax3.scatter([x2_data[-1]], [y2_data[-1]], color='blue')  # Highlight the latest point

        ax4.plot(wavelengths[0], np.mean(intensities,0), color=colors[i], label=f'Line {i+1}')

        # ax2.plot(intensities)
        plt.draw()
        plt.pause(0.1)

    focusedPos = z_axis_range[np.argmin(entropy_list)] 
    axis = c_uint(3)
    pos = c_double(focusedPos)

    # Move to a new position
    error = mcldll.MCL_SingleWriteN(pos, axis, handle)
    if error != 0:
            print("Error = ", error)
    # Wait for nanopositioner to settle
    time.sleep(0.025)
    # Read the new position
    position = mcldll.MCL_SingleReadN(axis, handle)
    
    print("Position = ", position)
    return (z_axis,entropy_graph)

##TO EDIT: filename inputs
date = "20230108"
filename = "plasma_60x_1mm_42mW_cw-853_grating3_quartz"

#Import MCL and initiate
# change the path to match your system.
mcldll = CDLL("C:/Users/yifei/Documents/NoodlePy/noodlepy/dlls/Madlib.dll") # need: "from ctypes import *" for this to work

# The correct return types for the functions used must be set.
mcldll.MCL_ReleaseHandle.restype = None
mcldll.MCL_SingleReadN.restype = c_double

# Acquire a device handle
handle = mcldll.MCL_InitHandle()
print("MCL Handle = ", handle)

#Initialize z axis position
axis = c_uint(3)
pos = c_double(0)

# Move to a new position
error = mcldll.MCL_SingleWriteN(pos, axis, handle)
print("Error = ", error)

# Wait for nanopositioner to settle
time.sleep(0.025)

# Read the new position
position = mcldll.MCL_SingleReadN(axis, handle)
print("Position = ", position)

bus = WasatchBus()
if not bus.device_ids:
    import sys
    
    print("no spectrometers found")
    sys.exit(1)

device_id = bus.device_ids[0]
print("found %s" % device_id)

device = WasatchDevice(device_id)
if not device.connect():
    print("connection failed")
    sys.exit(1)

print("connected to %s %s with %d pixels from (%.2f, %.2f)" % (
    device.settings.eeprom.model,
    device.settings.eeprom.serial_number,
    device.settings.pixels(),
    device.settings.wavelengths[0],
    device.settings.wavelengths[-1]))

print("setting integration time")
device.hardware.set_integration_time_ms(100)
# or: device.change_setting("integration_time_ms", 10)

print("Enabling laser")
device.hardware.set_laser_enable(True)

print(f"Waiting {5} sec for laser to warmup (required for MML)")
time.sleep(5)

autofocus(1000)

# for pixel in range(device.settings.pixels()):
    # print("%8.2f %8.2f" % (device.settings.wavelengths[pixel], spectrum[pixel]))

print("Disabling laser")
device.hardware.set_laser_enable(False)


plt.show()