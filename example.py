import cv2
import torch
from PIL import Image
import timeit
#print(torch.hub.get_dir())
CAM_API = cv2.CAP_MSMF

torch.hub.set_dir("./hub")
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').cuda()

# Images
for f in ['zidane.jpg', 'bus.jpg']:
    torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)  # download 2 images
img1 = Image.open('zidane.jpg')  # PIL image
img2 = cv2.imread('bus.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)
imgs = [img1, img2]  # batch of images

# Inference

loop = 300

#result = timeit.timeit('results = model(imgs, size=640)', globals=globals(), number=loop)
#print(result / loop)
  # includes NMS
results = model(imgs, size=640)
# Results
results.print()
results.show()
#results.save()  # or .show()
print(results)
print(results.xyxy[0])  # img1 predictions (tensor)
print(results.pandas().names)
print(results.pandas().xyxy[0])  # img1 predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
