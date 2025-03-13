#original source: https://github.com/13ff6/Topography_Map_Madagascar

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors


plt.close('all')

data = np.loadtxt('height.txt')
Long = data[:,0]; Lat = data[:,1]; Elev = data[:,2]

tl = 5
tw = 2
lw = 3
S = 30
pts=1000000


[x,y] =np.meshgrid(np.linspace(min(Long),max(Long),int(np.sqrt(pts))), np.linspace(min(Lat),max(Lat),int(np.sqrt(pts))))
z = griddata((Long, Lat), Elev, (x, y), method='linear')
x = np.matrix.flatten(x)
y = np.matrix.flatten(y)
z = np.matrix.flatten(z)


#########################################
#########TODO: FILL THIS FUNC############
def color_map(z):
	color = np.zeros((np.size(z),3))
	return color
#########################################
#########################################

###just a reference: you do not have to reproduce the same quality
#colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 56))
#colors_land = plt.cm.terrain(np.linspace(0.25, 1, 200))
#colors = np.vstack((colors_undersea, colors_land))
#cut_terrain_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_terrain', colors)
#plt.scatter(x,y,1,z,cmap = cut_terrain_map)

plt.scatter(x,y,s=1,c=color_map(z))

#plt.savefig("result.png",
            #bbox_inches ="tight",
            #pad_inches = 0)

plt.show()
