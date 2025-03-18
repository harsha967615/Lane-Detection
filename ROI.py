import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape[:2]  # Get image dimensions

    # Define a polygon for ROI dynamically
    polygon = np.array([
        [
            (int(width * 0.1), height),  # Bottom-left
            (int(width * 0.45), int(height * 0.6)),  # Top-left
            (int(width * 0.55), int(height * 0.6)),  # Top-right
            (int(width * 0.9), height)  # Bottom-right
        ]
    ], np.int32)

    # Create a black mask
    mask = np.zeros_like(img)

    # Create a filled polygonal mask
    cv2.fillPoly(mask, [polygon], 255)

    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contour on a blank mask
    contour_mask = np.zeros_like(img)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Apply the mask using bitwise AND
    masked_img = cv2.bitwise_and(img, contour_mask)

    return masked_img

# Load video
video_path = "E:\Project Work1\lane_video1.mp4"  # Change path as needed
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("End of video or cannot read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian Blur
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection
    roi_masked = region_of_interest(edges)  # Apply polygonal ROI masking

    cv2.imshow("Masked ROI (Polygonal Contour)", roi_masked)  # Show result

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
