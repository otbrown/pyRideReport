# RideReport.py
# Oliver Thomson Brown
# 2017-09-28

import gpxpy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as map_tiles

# READ IN GPX FILE
fpath = './GPX/Audax_Dick_McTaggart_s_150k_Classic.gpx'
f_gpx = open(fpath, 'r')
gpx = gpxpy.parse(f_gpx)
print('{} GPS Track loaded.'.format(gpx.tracks[0].name))
f_gpx.close()

# PULL USEFUL DATA
track_name = gpx.tracks[0].name
gps_track = gpx.tracks[0].segments[0].points
num_points = gpx.tracks[0].get_points_no()

# CREATE NUMERIC ARRAYS
dist = np.zeros(num_points - 1)
elevation = np.zeros(num_points)
dt = np.zeros(num_points - 1)
latitude = np.zeros(num_points)
longitude = np.zeros(num_points)

# FILL NUMERIC ARRAYS
for index in range(0, num_points-1):
    dist[index] = gps_track[index].distance_2d(gps_track[index+1])
    latitude[index] = gps_track[index].latitude
    longitude[index] = gps_track[index].longitude
    elevation[index] = gps_track[index].elevation
    dt[index] = (gps_track[index+1].time
                - gps_track[index].time).total_seconds()

elevation[num_points-1] = gps_track[num_points-1].elevation
latitude[num_points-1] = gps_track[num_points-1].latitude
longitude[num_points-1] = gps_track[num_points-1].longitude

# CALCULATE SECONDARY DATA
total_distance = np.append([0], dist.cumsum())
t_elapsed = np.append([0], dt.cumsum())
speed = np.append([0], (dist / dt))
avg_speed = np.append([0], (total_distance[1:] / t_elapsed[1:]))

# SMOOTH DATA
# define an approximate distance in metres over which to average speed
distance_window = 100
# roughly find the average distance between track points
avg_dist = dist.mean()
# define a window buffer that is the integer number of track points
# corresponding to approximately half the distance window
# then define the window size as twice that + 1, ensuring oddness
window_buf = np.around(distance_window / (2 * avg_dist), 0).astype(int)
window = 2 * window_buf + 1
smooth_speed = speed
smooth_elevation = elevation
for index in range(0, window_buf):
    start = 0
    end = index + window_buf + 1
    smooth_speed[index] = speed[start:end].mean()
    smooth_elevation[index] = elevation[start:end].mean()
for index in range(window_buf, num_points - window_buf):
    start = index - window_buf
    end = index + window_buf + 1
    smooth_speed[index] = speed[start:end].mean()
    smooth_elevation[index] = elevation[start:end].mean()
for index in range(num_points - window_buf, num_points - 1):
    start = index - window_buf
    end = num_points - 1
    smooth_speed[index] = speed[start:end].mean()
    smooth_elevation[index] = elevation[start:end].mean()

# CALCULATE CLIMB
dElev = np.diff(smooth_elevation)
total_climb = dElev[dElev>0].sum()

# CONVERT UNITS FOR PLOTTING
BRM_KMH = 15
METRES_IN_KM = 1000
SECONDS_IN_HOUR = 60 * 60
dist_km = total_distance / METRES_IN_KM
speed_kmh = smooth_speed * SECONDS_IN_HOUR / METRES_IN_KM
avg_kmh = avg_speed * SECONDS_IN_HOUR / METRES_IN_KM
brm_limit = BRM_KMH * np.ones(num_points)
hrs = np.floor(t_elapsed[-1] / SECONDS_IN_HOUR)
mins = np.around((t_elapsed[-1] - (hrs * SECONDS_IN_HOUR)) / 60)

# PLOTTING
# define colours
orange = (1, 69/255, 0)
bluegrey = (47/255, 79/255, 79/255)
plt.style.use('seaborn-darkgrid')

fig = plt.figure()
fig.suptitle(track_name, fontsize=16, fontweight='bold')
gs = mpl.gridspec.GridSpec(2,2)

osm_tiles = map_tiles.OSM()
ax0 = plt.subplot(gs[0:2,0], projection=osm_tiles.crs)
ax0.plot(longitude, latitude, transform=ccrs.PlateCarree(),
         color='red', linewidth=2)
ax0.add_image(osm_tiles, 12)
notes = 'Distance: {:.2f}km\nTime: {:g}hrs {:g}mins\nClimb: {:g}m'.format(dist_km[-1], hrs, mins, total_climb)
ax0.text(0, -0.1, notes, transform=ax0.transAxes, fontsize=14)

ax1 = plt.subplot(gs[0,1])
ax1.plot(dist_km, speed_kmh, color=orange, linewidth=0.6)
ax1.plot(dist_km, avg_kmh, '--', linewidth=2)
ax1.plot(dist_km, brm_limit, ':k', linewidth=2)
ax1.set_xlabel('Distance (km)')
ax1.set_ylabel('Speed (km/h)')
ax1.legend(['Speed', 'Average Speed', 'BRM Limit'])
ax1.autoscale(tight=True)

ax2 = plt.subplot(gs[1,1])
ax2.fill_between(dist_km, 0, smooth_elevation, color=bluegrey, alpha=0.5)
ax2.set_xlabel('Distance (km)')
ax2.set_ylabel('Elevation (m)')
ax2.autoscale(tight=True)

plt.show()
