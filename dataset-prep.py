import cv2

# Open webcam index 1
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam with index 1")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Webcam 1", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
