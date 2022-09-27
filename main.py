import cv2

# load the image
nucleus = cv2.imread('images/figure2.png', cv2.IMREAD_COLOR)

# sets the interval of the color to be detected
start_interval = (98,53,100)
end_interval = (150,81,100)

# generate a mask
mask = cv2.inRange(nucleus, (100, 0, 0), (255, 80, 80))
cv2.imwrite('results/generated-mask.png', mask)

# recognize patterns in the image using the mask 
blur = cv2.GaussianBlur(mask, (5,5), 10)
thresh = cv2.adaptiveThreshold(blur, 20, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilate = cv2.dilate(mask, kernel, iterations=1)

# count all white objects in image and draw a red rectangle around them
contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# check all occurrences
i=1
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(nucleus, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(nucleus, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, i)
    i+=1

cv2.imshow('nucleus', nucleus)
cv2.imwrite('results/result.png', nucleus)


