import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

def distortion_coefs( nx=9, ny=6 ):
    imgpoints = []
    objpoints = []

    objp = np.zeros( (nx*ny,3), np.float32 )
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    for i in range(1,21):
        filename = "camera_cal/calibration" + str(i) + ".jpg"
        img = mpimg.imread(filename)

        gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )

        ret, corners = cv2.findChessboardCorners( gray, (nx,ny), None )

        print(ret)

        if ret == True:
            imgpoints.append( corners )
            objpoints.append( objp )
            
            #draw the corners on the image
            img = cv2.drawChessboardCorners( img, (nx,ny), corners, ret )

    return cv2.calibrateCamera( objpoints, imgpoints, gray.shape[::-1], None, None )

# Apply filters to locate suspected lane line pixels.
# The final binary output is produced by combining them as follows:
# lightness filter & ( ( gradient magnitude filter & gradient angle filter ) | saturation filter ).
def apply_filters( img, sobel_kernel=3, 
                   thresh_lightness=40,
                   thresh_angle=(0,np.pi/2),
                   thresh_magnitude=(0,255),
                   thresh_saturation=(0,255) ):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # x and y gradients using Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_x = np.abs( sobel_x )
    abs_y = np.abs( sobel_y )
    # Angle of absolute value of x and y gradients between 0 and pi/2
    angles = np.arctan2( abs_y, abs_x )
    # Compute magnitude of gradient
    magnitude = np.sqrt(sobel_x**2+sobel_y**2)
    scaled_magnitude = np.uint8( 255*magnitude/np.max( magnitude ) )

    # Create arrays of lightness and saturation values
    hls = cv2.cvtColor( img, cv2.COLOR_RGB2HLS )
    lightness = hls[:,:,1]
    saturation = hls[:,:,2]

    # 1s for possible lane line pixels
    binary_output = np.zeros_like( scaled_magnitude )
    binary_output[ ( lightness > thresh_lightness ) &
                   ( ( ( scaled_magnitude >= thresh_magnitude[0] ) &  filter
                     ( scaled_magnitude <= thresh_magnitude[1] ) &
                     ( angles >= thresh_angle[0] ) &
                     ( angles <= thresh_angle[1] ) ) |
                     ( ( saturation >  thresh_saturation[0] ) &
                     ( saturation <= thresh_saturation[1] ) ) ) ] = 1

    return scaled_magnitude, binary_output

# Window search
width_window = 150
height_window = 180
margin = 100

# Array of 1s where a rectangular mask is present
def mask_window(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

# Find centroids by taking considering horizontal slices of the array.
# Pixels are summed vertically, and the 1D array is convolved with a mask
# This convolution should have 2 peaks corresponding to the centers of the
# left and right lane lines in that horizontal slice.

def find_window_centroids(warped, width_window, height_window, margin):
    
    window_centroids = []
    window = np.ones(width_window)
    
    # Add bottom of image to get slice
    left_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    left_center = np.argmax(np.convolve(window,left_sum))-width_window/2
    right_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    right_center = np.argmax(np.convolve(window,right_sum))-width_window/2+int(warped.shape[1]/2)
    
    window_centroids.append((left_center,right_center))
    
    # Look for max pixel locations
    for level in range(1,(int)(warped.shape[0]/height_window)):
	    # Vertically sum this horizontal slice of the image
	    image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*height_window):int(warped.shape[0]-level*height_window),:], axis=0)
	    convolve_signal = np.convolve(window, image_layer)
	    # Find left centroid
	    # Use width_window/2 as offset
	    offset = width_window/2
	    left_min_index = int(max(left_center+offset-margin,0))
	    left_max_index = int(min(left_center+offset+margin,warped.shape[1]))
	    left_center = np.argmax(convolve_signal[left_min_index:left_max_index])+left_min_index-offset
	    # Find right centroid
	    right_min_index = int(max(right_center+offset-margin,0))
	    right_max_index = int(min(right_center+offset+margin,warped.shape[1]))
	    right_center = np.argmax(convolve_signal[right_min_index:right_max_index])+right_min_index-offset

	    window_centroids.append((left_center,right_center))

    return window_centroids

# Returns coefficients of quadratic polynomial fitted to left and right lane lines
def fit_lane_lines( warped, preexistingFit=False, left_fit=None, right_fit=None ):
    
    # RGB of grayscale warped image
    warped_rgb = np.array(cv2.merge((warped,warped,warped)),np.uint8)
   
    # If polynomial fit is false..
    # Sliding window search to find lane line pixels
    # Fit a polynomial to those pixels.
    if( preexistingFit == False ): 
        window_centroids = find_window_centroids(warped, width_window, height_window, margin)
        
        # Window centers = true..
        if len(window_centroids) > 0:
        
            # Points used to draw windows
            left_points = np.zeros_like(warped)
            right_points = np.zeros_like(warped)
        
            # Draw the windows on each level
            for level in range(0,len(window_centroids)):
                # mask_window is a function to draw window areas
        	    left_mask = mask_window(width_window,height_window,warped,window_centroids[level][0],level)
        	    right_mask = mask_window(width_window,height_window,warped,window_centroids[level][1],level)
        	    # Add graphic points from window mask here to total pixels found 
        	    left_points[(left_points == 255) | ((left_mask == 1) ) ] = 255
        	    right_points[(right_points == 255) | ((right_mask == 1) ) ] = 255
        
        
            # add left and right window pixels, create zero color channel, green window, overlay with window results
            template = np.array(right_points+left_points,np.uint8)
            zero_channel = np.zeros_like(template)
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8)
            output = cv2.addWeighted(warped_rgb, 0.5, template, 0.5, 0.0)
         
        else:
            output = warped_rgb

        left_pixels_to_fit = np.zeros_like(warped)
        right_pixels_to_fit = np.zeros_like(warped)

        left_pixels_to_fit[( warped > 0 ) & ( left_points > 0 )] = 1
        right_pixels_to_fit[( warped > 0 ) & ( right_points > 0 )] = 1

        # xy values of left and right lane lines
        left_x_values_to_fit = left_pixels_to_fit.non_zero()[1]
        left_y_values_to_fit = left_pixels_to_fit.non_zero()[0]
        right_x_values_to_fit = right_pixels_to_fit.non_zero()[1]
        right_y_values_to_fit = right_pixels_to_fit.non_zero()[0]

        # Fit a quadratic to the left and right pixel locations, with y as the "x-coordinate" of the polynomial.
        left_fit = np.polyfit( left_y_values_to_fit, left_x_values_to_fit, 2)
        right_fit = np.polyfit( right_y_values_to_fit, right_x_values_to_fit, 2)

    # If a polynomial fit lane lines were found in the last frame, search near them for lane lines
    # instead of redoing the sliding window search.
    # 
    # I did not end up using this in my pipeline, because it worked on the first try
    # using only sliding windows
    else:
        non_zero = warped.non_zero()
        non_zero_x = np.array(non_zero[1])
        non_zero_y = np.array(non_zero[0])

        left_lane_inds = ((non_zero_x > (left_fit[0]*(non_zero_y**2) + left_fit[1]*non_zero_y + left_fit[2] - margin)) &
                          (non_zero_x < (left_fit[0]*(non_zero_y**2) + left_fit[1]*non_zero_y + left_fit[2] + margin)))
        right_lane_inds = ((non_zero_x > (right_fit[0]*(non_zero_y**2) + right_fit[1]*non_zero_y + right_fit[2] - margin)) &
                           (non_zero_x < (right_fit[0]*(non_zero_y**2) + right_fit[1]*non_zero_y + right_fit[2] + margin)))
        
        # Extract left and right line pixel positions
        x_left = non_zero_x[left_lane_inds]
        y_left = non_zero_y[left_lane_inds]
        x_right = non_zero_x[right_lane_inds]
        y_right = non_zero_y[right_lane_inds]

        left_fit = np.polyfit(y_left, x_left, 2)
        right_fit = np.polyfit(y_right, x_right, 2)

    return left_fit, right_fit, output

# Draw the lane region undistorted image
def draw_unwarped_lane_region( warped, undist, Minv, left_fit, right_fit ):
    # Draw lines on output image
    zero_warp = np.zeros_like(warped).astype(np.uint8)
    warp_color = np.dstack((zero_warp, zero_warp, zero_warp))

    y_plot = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
 
    # Create a list of x-points from the y-points and the polynomial fits
    fitx_x_left = left_fit[0]*y_plot**2 + left_fit[1]*y_plot + left_fit[2]
    fitx_x_right = right_fit[0]*y_plot**2 + right_fit[1]*y_plot + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([fitx_x_left, y_plot]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fitx_x_right, y_plot])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(warp_color, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(warp_color, Minv, (warped.shape[1], warped.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # plt.imshow(result)
    # plt.show()

    return result


# Use the best-fit lines in pixel space to get best-fit lines in real-world dimensions
# This function is based on code from the lessons.
def get_radii_of_curvature( img, left_fit, right_fit ):

    # meters per pixel
    y_meters_per_pixel = 30/720
    x_meters_per_pixel = 3.7/700

    # y-points list
    y_plot = np.linspace(0, img.shape[0]-1, img.shape[0] )
 
    # Create list of x-points in pixels space
    fitx_x_left = left_fit[0]*y_plot**2 + left_fit[1]*y_plot + left_fit[2]
    fitx_x_right = right_fit[0]*y_plot**2 + right_fit[1]*y_plot + right_fit[2]

    evaluate_y = img.shape[0]

    # xy in world space
    left_fit_cr = np.polyfit(y_plot*y_meters_per_pixel, fitx_x_left*x_meters_per_pixel, 2)
    right_fit_cr = np.polyfit(y_plot*y_meters_per_pixel, fitx_x_right*x_meters_per_pixel, 2)
    # Radius in meters
    left_radius = ((1 + (2*left_fit_cr[0]*evaluate_y*y_meters_per_pixel + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_radius = ((1 + (2*right_fit_cr[0]*evaluate_y*y_meters_per_pixel + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_radius, right_radius

