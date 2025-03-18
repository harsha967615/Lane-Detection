import cv2
import numpy as np

# Load video
video_path = "E:\Project Work1/lane_video1.mp4"  # Update as needed
cap = cv2.VideoCapture(video_path)

def process_frame(frame):
    """Process a single frame to detect lane lines."""
    
    # Step 1: Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply Gaussian Blur (reduces noise)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3: Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # DEBUG: Show Edge Detection Output
    cv2.imshow("Edges", edges)

    # Step 4: Define Polygonal ROI
    height, width = edges.shape
    mask = np.zeros_like(edges)
    roi_polygon = np.array([[
        (int(width * 0.2), height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.8), height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_polygon, 255)

    # Apply Mask
    masked_edges = cv2.bitwise_and(edges, mask)

    # DEBUG: Show ROI Mask Output
    cv2.imshow("ROI Masked", masked_edges)

    # Step 5: Apply Hough Line Transform
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 50, minLineLength=50, maxLineGap=200)
    
    # Step 6: Draw Detected Lines
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 8)  # Yellow thick lines

    # Combine with Original Frame
    output = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    return output

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)
    cv2.imshow("Lane Detection", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
