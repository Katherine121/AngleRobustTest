# the coordinates of the end point which is masked to satisfy double-blind principle
# the start point
START_LAT = 30.3093714
START_LON = 119.9232729
# the end point
END_LAT = 30.2620077
END_LON = 119.947846
# the kinds of disturbances
NOISE_DB = [["ori", "ori"], ["ori", "random"], ["ori", "uni"], ["ori", "hei"],
            ["cutout", "ori"], ["rain", "ori"], ["snow", "ori"], ["fog", "ori"],
            ["bright", "ori"],
            ]
# the random position shift of the start point
START_SHIFT = 25
# the horizontal position deviation
# random wind
RANDOM = 5
# one-way wind
UNI = 1
# altitude
HEIGHT_SHIFT = 40
# cutout
SIZE = 0.2
# RANDOM = 10
# UNI = 1.5
# HEIGHT_SHIFT = 60
# SIZE = 0.4
