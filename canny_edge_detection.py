import cv2

# Open the video file
cap = cv2.VideoCapture("lane_video1.mp4")  # Replace "video.mp4" with your actual video file

# Check if the video file opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    # Read a frame from the video
    ret, img = cap.read()

    # If no frame is read (end of video), break the loop
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image
    kernel_size = 5  # Kernel size for blurring
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Apply Canny edge detection
    low_threshold = 50  # Lower threshold for edge detection
    high_threshold = 150  # Higher threshold for edge detection
    edges = cv2.Canny(blur, low_threshold, high_threshold)

    # Display the edge-detected frame
    cv2.imshow("Canny Edge Detection", edges)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close windows
cap.release()
cv2.destroyAllWindows()
