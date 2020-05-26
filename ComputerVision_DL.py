
# ------------------------------------- Code for Opening webcam -------------------------------------
# Import Libraries
import cv2

cam_capture = cv2.VideoCapture(0)
cv2.destroyAllWindows()
    
while True:   
    _, image_frame = cam_capture.read()    
    
    sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)    
    
    cv2.imshow("Sketcher ROI", image_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cam_capture.release()
cv2.destroyAllWindows()

