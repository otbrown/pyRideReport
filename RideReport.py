# RideReport.py
# Oliver Thomson Brown
# 2017-09-28

import ridereport as rr

ride = rr.RideReport()
ride.gpxload('GPX/Audax_Dick_McTaggart_s_150k_Classic.gpx')
ride.analyse()
ride.brevet(150, 15)
ride.report()
