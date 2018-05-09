import cv2

path = r"F:\00_gitbio\OCR_bio\OCR_bio\data\test.png"
path1 = r"F:\00_gitbio\OCR_bio\OCR_bio\data\test2.png"

img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)
imgc = cv2.imread(path)

_ , contours , _ = cv2.findContours( img , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
rectList = []

cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
for i in range(len(contours)):
    cv2.drawContours(imgc , contours ,  i , (0,255,0) , 3 )

cv2.imwrite(path1 , imgc)

for contour in contours:

    rect = Rectangle( cv2.boundingRect( contours ) )

    if rect.area < max and rect.area > min:
        rectList.append( rect )


print()