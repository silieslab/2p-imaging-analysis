
import glob
import os
import tifffile
import matplotlib.pyplot as plt

dir = input(">>> Copy-paste directory string containing the single tif files:")
os.chdir(dir)

files = glob.glob(dir+'\\'+'*ome.tif')
filename = files[0]
#Seb: The next line is not clear to me but just taking the first image (e.g, 00001.ome.tif) is saving all *ome.tif files. as stack
tifffile.imsave('Test-TSeries-stack.tif', tifffile.imread(filename), bigtiff=True)

# TSeb, the following lines are saving a stack double the length as it should. Commented out
# with tifffile.TiffWriter('TSeries-stack.tif') as stack:
#     for filename in glob.glob(dir+'\\'+'*ome.tif'):
#         stack.save(tifffile.imread(filename),photometric='minisblack',contiguous=False) # 
        
# print('TIF-stack saved')