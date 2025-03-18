import cv2
import numpy as np

# This function finds edges in the image using the Canny Edge Detection method.
def canny(img):
    # If there is no image (like if there's an error with the video), close everything and exit
    if img is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    
    # Convert the image to shades of gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Set the size of the blur to make the image smoother
    kernel = 5
    
    # Apply a blur to the gray image to remove noise (unwanted details)
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    
    # Use the Canny function to find edges in the blurred image
    canny = cv2.Canny(blur, 50, 150)
    
    # Return the edges found
    return canny

# This function creates a mask to focus on a specific area of the image (the road).
def region_of_interest(canny):
    # Get the height and width of the image
    height = canny.shape[0]
    width = canny.shape[1]
    
    # Make a black image (same size as the original) to create the mask
    mask = np.zeros_like(canny)
    
    # Define a triangle shape where we want to keep the image (the rest will be blacked out)
    triangle = np.array([[
        (int(width * 0.1), height),          # Bottom-left corner of the triangle
        (int(width * 0.5), int(height * 0.6)), # Top-center corner of the triangle
        (int(width * 0.9), height),          # Bottom-right corner of the triangle
    ]], np.int32)
    
    # Fill in the triangle with white, so we can see through it
    cv2.fillPoly(mask, triangle, 255)
    
    # Use the triangle mask to keep only the region inside it and black out the rest
    masked_image = cv2.bitwise_and(canny, mask)
    
    # Return the masked image (only the region of interest is visible)
    return masked_image

# This function finds lines in the image using a method called Hough Transform.
def houghLines(cropped_canny):
    # Find lines in the image based on edges and return them
    return cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

# This function combines the original image and the image with drawn lines.
def addWeighted(frame, line_image):
    # Merge the original image and the line image with a blend effect
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

# This function draws lines on an empty image of the same size as the input image.
def display_lines(img, lines):
    # Create an empty black image of the same size as the input image
    line_image = np.zeros_like(img)
    
    # Check if any lines are detected
    if lines is not None:
        # Loop through each detected line
        for line in lines:
            # Draw each line with red color and thickness of 10
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    
    # Return the image with lines drawn on it
    return line_image

# This function calculates points to draw lane lines.
def make_points(image, line):
    # Get the slope and intercept of the line
    slope, intercept = line
    
    # Set the start and end y-coordinates for the line
    y1 = int(image.shape[0])              # Bottom of the image
    y2 = int(y1 * 3.0 / 5)                # Slightly above the center
    
    # Calculate the start and end x-coordinates based on the line equation y = mx + b
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    # Return the coordinates of the line to be drawn
    return [[x1, y1, x2, y2]]

# This function calculates the average slope and intercept for left and right lane lines.
def average_slope_intercept(image, lines):
    # Lists to hold the lines on the left and right side
    left_fit = []
    right_fit = []

    # If no lines are detected, return nothing
    if lines is None:
        return None
    
    # Loop through each line
    for line in lines:
        # Get the x and y coordinates for the start and end of the line
        for x1, y1, x2, y2 in line:
            try:
                # Calculate the slope and intercept of the line
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                
                # Skip lines that are almost vertical
                if abs(slope) < 0.5:
                    continue
                
                # Separate lines into left and right based on the slope
                if slope < 0:
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))
            except np.RankWarning:
                # Skip if it fails to fit the line
                continue
    
    # Calculate the average slope and intercept for left and right lines
    left_line = make_points(image, np.average(left_fit, axis=0)) if left_fit else None
    right_line = make_points(image, np.average(right_fit, axis=0)) if right_fit else None

    # List to hold the final left and right lines
    averaged_lines = []
    if left_line is not None:
        averaged_lines.append(left_line)
    if right_line is not None:
        averaged_lines.append(right_line)
    
    # Return the left and right lane lines
    return averaged_lines

# Start capturing the video from file
print("started")
cap = cv2.VideoCapture("lane_video2.mp4")
print(cap.isOpened())

# Go through the video, frame by frame
while cap.isOpened():
    # Read one frame from the video
    _, frame = cap.read()
    
    # If the frame is empty (end of video), stop the loop
    if frame is None:
        break
    
    # Find the edges in the current frame
    canny_image = canny(frame)
    
    # Apply the mask to focus on the road area
    cropped_canny = region_of_interest(canny_image)
    
    # Find the lines in the road area
    lines = houghLines(cropped_canny)
    
    # Calculate the average lane lines from the detected lines
    averaged_lines = average_slope_intercept(frame, lines)
    
    # Draw the average lane lines on a black image
    line_image = display_lines(frame, averaged_lines)
    
    # Combine the original frame with the lane lines image
    combo_image = addWeighted(frame, line_image)
    
    # Display the final image with lane lines
    cv2.imshow("result", combo_image)
    
    # If the 'q' key is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print "stop" when finished
print("stop")

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()