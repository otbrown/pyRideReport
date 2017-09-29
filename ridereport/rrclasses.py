# rrclasses.py
# class definition file
# Oliver Thomson Brown
# 2017-09-29

import gpxpy
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as map_tiles

class RideReport:
    def __init__(self):
        self.name = None
        self.latitude = None
        self.longitude = None
        self.elevation = None
        self.diff_distance = None
        self.diff_time = None
        self.distance = None
        self.elapsed_time = None
        self.speed = None
        self.avg_speed = None
        self.smooth_speed = None
        self.smooth_elevation = None
        self.total_time = None
        self.total_climb = None
        self.isBrevet = False
        self.brevet_distance = None
        self.brevet_speed = None
        self.time_in_hand = None

    def gpxload(self, fpath):
        '''
        Loads a gpx file and stores the primary gps data in the
        RideReport object.

        Populates the following attributes:
            name
            latitude
            longitude
            elevation
            diff_distance
            diff_time

        fpath should be a python "path-like object"
        '''
        f_gpx = open(fpath, 'r')
        gpx = gpxpy.parse(f_gpx)
        f_gpx.close()

        self.name = gpx.tracks[0].name
        num_points = gpx.tracks[0].get_points_no()
        gps_track = gpx.tracks[0].segments[0].points

        self.latitude = np.zeros(num_points)
        self.longitude = np.zeros(num_points)
        self.elevation = np.zeros(num_points)
        self.diff_distance = np.zeros(num_points - 1)
        self.diff_time = np.zeros(num_points - 1)

        for index in range(0, num_points-1):
            self.latitude[index] = gps_track[index].latitude
            self.longitude[index] = gps_track[index].longitude
            self.elevation[index] = gps_track[index].elevation
            self.diff_distance[index] = (gps_track[index]
                                        .distance_2d(gps_track[index+1])
                                        )
            self.diff_time[index] = ((gps_track[index+1].time
                                    - gps_track[index].time)
                                    .total_seconds())

        self.latitude[-1] = gps_track[-1].latitude
        self.longitude[-1] = gps_track[-1].longitude
        self.elevation[-1] = gps_track[-1].elevation

    def analyse(self):
        '''
        Calculates secondary data from the already loaded raw GPS
        information.

        Populates the following attributes:
            distance
            elapsed_time
            total_time
            speed
            avg_speed
            smooth_speed
            smooth_elevation
            total_climb

        Default distance window for smoothing is 100m.
        '''
        self.distance = np.append([0], self.diff_distance.cumsum())
        self.elapsed_time = np.append([0], self.diff_time.cumsum())
        self.total_time = self.elapsed_time[-1]
        self.speed = np.append([0],
                               (self.diff_distance / self.diff_time))
        self.avg_speed = np.append([0],
                                   (self.distance[1:]
                                   / self.elapsed_time[1:]))

        # smooth speed and elevation data
        distance_window = 100
        avg_dist = self.diff_distance.mean()
        window = np.around(distance_window / avg_dist, 0).astype(int)
        self.smooth_speed = movavg(self.speed, window)
        self.smooth_elevation = movavg(self.elevation, window)

        diff_elevation = np.diff(self.smooth_elevation)
        self.total_climb = diff_elevation[diff_elevation > 0].sum()

    def brevet(self, brevet_distance, brevet_speed):
        '''
        Sets RideReport.isBrevet to true, passes through supplied
        brevet_distance and brevet_speed, and calculates time in hand

        Requires self.elapsed_time to have been calculated

        NOTE: assumes distance in km, and speed in km/h!
        '''
        self.isBrevet = True
        self.brevet_distance = brevet_distance
        self.brevet_speed = brevet_speed

        SECONDS_IN_HOUR = 60 * 60
        METRES_IN_KM = 1000
        time_limit = SECONDS_IN_HOUR * brevet_distance / brevet_speed
        delta_time = time_limit - self.elapsed_time
        delta_distance = ((self.distance[-1] - self.distance)
                          / METRES_IN_KM)
        self.time_in_hand = ( delta_time - (SECONDS_IN_HOUR *
                            (delta_distance / brevet_speed)) )

    def delete_brevet(self):
        '''
        Undo a RideReport.brevet() call, in case a mistake is made. If
        you just need to change the brevet parameters (brevet_speed or
        brevet_distance), a second call to RideReport.brevet() will do.
        '''
        self.isBrevet = False
        self.brevet_distance = None
        self.brevet_speed = None
        self.time_in_hand = None

    def report(self):
        '''
        Generates a figure showing the a map including the GPS track,
        speed data, elevation data, and if brevet information is
        supplied, time in hand.
        '''

        # unit conversions
        METRES_IN_KM = 1000
        SECONDS_IN_HOUR = 60 * 60
        SECONDS_IN_MIN = 60
        distance_km = self.distance / METRES_IN_KM
        speed_kmh = self.smooth_speed * SECONDS_IN_HOUR / METRES_IN_KM
        avg_kmh = self.avg_speed * SECONDS_IN_HOUR / METRES_IN_KM
        hrs = np.floor(self.total_time / SECONDS_IN_HOUR)
        mins = np.around((self.total_time - (hrs * SECONDS_IN_HOUR))
                         / SECONDS_IN_MIN)

        # style and colour information
        orange = (1, 69/255, 0)
        bluegrey = (47/255, 79/255, 79/255)
        plt.style.use('seaborn-darkgrid')

        # plotting
        fig = plt.figure()
        fig.suptitle(self.name, fontsize=16, fontweight='bold')
        if self.isBrevet:
            gs = gridspec.GridSpec(3,2)
        else:
            gs = gridspec.GridSpec(2,2)

        osm_tiles = map_tiles.OSM()
        ax0 = plt.subplot(gs[:,0], projection=osm_tiles.crs)
        ax0.plot(self.longitude, self.latitude,
                 transform=ccrs.PlateCarree(), color='red', linewidth=2)
        ax0.add_image(osm_tiles, 12)
        notes = 'Distance: {:.2f}km\nTime: {:g}hrs {:g}mins\nClimb: {:.2f}m'.format(distance_km[-1], hrs, mins, self.total_climb)
        ax0.text(0, -0.1, notes, transform=ax0.transAxes, fontsize=14)

        ax1 = plt.subplot(gs[0,1])
        ax1.plot(distance_km, speed_kmh, color=orange, linewidth=0.6)
        ax1.plot(distance_km, avg_kmh, '--', linewidth=2)
        if self.isBrevet:
            ax1.plot(distance_km, (self.brevet_speed
                     * np.ones(len(self.distance))), ':k', linewidth=2)
        ax1.set_xlabel('Distance (km)')
        ax1.set_ylabel('Speed (km/h)')
        if self.isBrevet:
            ax1.legend(['Speed', 'Average Speed', 'Brevet Limit'])
        else:
            ax1.legend(['Speed', 'Average Speed'])
        ax1.autoscale(tight=True)

        ax2 = plt.subplot(gs[1,1])
        ax2.fill_between(distance_km, 0, self.smooth_elevation,
                         color=bluegrey, alpha=0.5)
        ax2.set_xlabel('Distance (km)')
        ax2.set_ylabel('Elevation (m)')
        ax2.autoscale(tight=True)

        if self.isBrevet:
            ax3 = plt.subplot(gs[2,1])
            ax3.fill_between(distance_km, 0, self.time_in_hand)
            ax3.set_xlabel('Distance (km)')
            ax3.set_ylabel('Time in Hand (s)')
            ax3.autoscale(tight=True)

        plt.show()

def movavg(data, window):
    '''
    Calculates a moving average using a particular window size.
    The window is truncated when it meets the edge of the data.

    data is a 1d numpy array, window is an integer
    if window is odd, then it defines a region centred on the current
    index, over which the values are averaged
    if window is even it defines the same region as window + 1 would
    '''
    N = len(data)
    window_buffer = np.floor(window / 2).astype(int)
    smooth_data = data
    for index in range(0, window_buffer):
        start = 0
        end = index + window_buffer + 1
        smooth_data[index] = data[start:end].mean()
    for index in range(window_buffer, N - window_buffer):
        start = index - window_buffer
        end = index + window_buffer + 1
        smooth_data[index] = data[start:end].mean()
    for index in range(N - window_buffer, N - 1):
        start = index - window_buffer
        end = N - 1
        smooth_data[index] = data[start:end].mean()

    return smooth_data
