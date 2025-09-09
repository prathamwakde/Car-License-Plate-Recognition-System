import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

image =cv2.imread('car.jpg')
cv2.imshow("Original Image",image)

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image",gray_image)

canny_edge = cv2.Canny(gray_image,170,200)
cv2.imshow("Canny Edge",canny_edge)

contours, new = cv2.findContours(canny_edge.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours,key=cv2.contourArea,reverse=True)[:150]

contours_with_license_plate = None
license_plate = None
x,y,w,h = None,None,None,None

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
    if len(approx) == 4:
        # Check aspect ratio
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if aspect_ratio >= 2.0 and aspect_ratio <= 5.0:  # Typical aspect ratio
            contours_with_license_plate = approx
            license_plate = gray_image[y:y+h, x:x+w]
            break 
(thresh, license_plate) = cv2.threshold(license_plate,150,180,cv2.THRESH_BINARY)
cv2.imshow("License Plate",license_plate)

license_plate = cv2.bilateralFilter(license_plate,11,17,17)

text = pytesseract.image_to_string(license_plate)

image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
image = cv2.putText(image,text,(x-100,y-50),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),6,cv2.LINE_AA)

print("License Plate :",text)
cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

