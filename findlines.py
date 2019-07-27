# For processing video files
from moviepy.editor import VideoFileClip

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import functions as fn

# Distortion coefficients
ret, mtx, dist, rvecs, tvecs  = fn.distortion_coefs()

def pipeline(img):
    # Undistort image
    undist = cv2.undistort( img, mtx, dist, None, mtx )

    x_size = undist.shape[1]
    y_size = undist.shape[0]
    img_size = [x_size, y_size]

    # Apply thresholds, return binary image
    S, filtered = fn.apply_filters( undist,
                                    thresh_lightness=40,
                                    thresh_angle=(0.8,1.2),
                                    thresh_magnitude=(40,255), 
                                    thresh_saturation=(100,255)  )

    # Source points
    source_top_left = [592,450]
    source_top_right = [692,450]
    source_bottom_right = [1121,y_size]
    source_bottom_left = [209,y_size]

    source = np.float32( [source_bottom_right, source_bottom_left, source_top_left, source_top_right] )

    # Destination points
    destination_bottom_right = [source_bottom_right[0],y_size]
    destination_bottom_left = [source_bottom_left[0],y_size]
    destination_top_left = [source_bottom_left[0],0]
    destination_top_right = [source_bottom_right[0],0]
    destination = np.float32( [destination_bottom_right, destination_bottom_left, destination_top_left, destination_top_right] )

    # Perspective transform matrix
    M = cv2.getPerspectiveTransform(source, destination)
    # Inverse transform
    Minv = cv2.getPerspectiveTransform(destination, source)
    # Warp and grayscale image
    warped = 255*cv2.warpPerspective( filtered, M, (x_size,y_size), flags=cv2.INTER_LINEAR )

    # Left and right parabolas fitted to lane lines
    left_fit, right_fit, output = fn.fit_lane_lines( warped )
  
    # Draw the lane line region between the two fitted parabolas on the undistorted frame.
    unwarped_lane_region = fn.draw_unwarped_lane_region( warped, undist, Minv, left_fit, right_fit )
    
    # Radius of curvature
    rl, rr = fn.get_radii_of_curvature( unwarped_lane_region, left_fit, right_fit )

    # Print on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText( unwarped_lane_region, "Left radius of curvature = {:4.1f} m".format(rl),
                 (200,100), font, 1.2, (255,255,255), 2, cv2.LINE_AA )
    cv2.putText( unwarped_lane_region, "Right radius of curvature = {:4.1f} m".format(rr),
                 (200,160), font, 1.2, (255,255,255), 2, cv2.LINE_AA )

    # Distance (pixels) off center
    
    # Bottom y-coordinates
    y_plot = img.shape[0]-1
    # Bottom x-coordinates
    x_left = left_fit[0]*y_plot**2 + left_fit[1]*y_plot + left_fit[2]
    x_right = right_fit[0]*y_plot**2 + right_fit[1]*y_plot + right_fit[2]
    distance_off_center_pixels = np.float64( unwarped_lane_region.shape[1] )/2. - (x_right+x_left)/2.
    x_meters_per_pixel = 3.7/700
    # World space
    distance_off_center_meters = x_meters_per_pixel*distance_off_center_pixels

    if( distance_off_center_meters > 0 ):
      cv2.putText( unwarped_lane_region, "Position = {:3.2f} m right of center".format(distance_off_center_meters),
		   (200,220), font, 1.2, (255,255,255), 2, cv2.LINE_AA )
    else:
      cv2.putText( unwarped_lane_region, "Position = {:3.2f} m left of center".format(np.abs(distance_off_center_meters)),
		   (200,220), font, 1.2, (255,255,255), 2, cv2.LINE_AA )

    return unwarped_lane_region

# Open
clip = VideoFileClip('project_video.mp4')
# Create
output_clip = clip.fl_image( pipeline )
# Write
output_clip.write_videofile( 'project_output.mp4', audio=False)
