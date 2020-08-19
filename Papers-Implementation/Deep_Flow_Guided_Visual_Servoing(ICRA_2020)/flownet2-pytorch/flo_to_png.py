import flowiz as fz
import cv2
import glob

files = glob.glob('result.flo')
img = fz.convert_from_file(files[0])
cv2.imwrite('result.png', img)
