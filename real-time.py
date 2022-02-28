import cv2
import torch
from PIL import Image
import PyQt5
from PyQt5 import Qt
from PyQt5 import QtGui,QtWidgets
from PyQt5.QtWidgets import QMainWindow,QApplication
from PyQt5.QtCore import QRect,QPoint,Qt
from PyQt5.QtGui import QPainter,QPen,QPixmap
import sys

#######################################
# VideoCapture Init
#######################################
print("VideoCapture Init")

VC = cv2.VideoCapture(0)
chk =  False
frame = None
while not chk:
    chk , frame = VC.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#assert frame

##########################################
# Model Init
#######################################
print("Model Init")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').cuda()

results = model(frame, size=640)

NAME_LIST = results.pandas().names

##############################
# Dummy value
"""
print("Load dummy value")
for f in ['bus.jpg']:
    torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)  # download 2 images
dummy_img = cv2.imread('bus.jpg')
dummy_img = cv2.cvtColor(dummy_img, cv2.COLOR_BGR2RGB)

dummy_batch = [dummy_img]

dummy_res = model(dummy_batch, size=640).xyxy[0]
"""
#

def convert_cv2qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    #rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    #p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
    return QPixmap.fromImage(convert_to_Qt_format)

def unPackRes(res:torch.FloatTensor):
    return res.cpu().numpy().tolist()

class ShowResLabel(QtWidgets.QLabel):
    rectPos = []
    NewFrameReady = False
    frame = None
    #Mouse click event
    def SetFrame(self,res,frame):
        self.rectPos = unPackRes(res.cpu())
        self.frame = frame
        self.NewFrameReady = True
    """
    def mousePressEvent(self,event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()
        #Mouse release event
    def mouseReleaseEvent(self,event):
        self.flag = False
        #Mouse movement events
    def mouseMoveEvent(self,event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            #self.update()
            #Draw events
    """
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.NewFrameReady:

            pixmap = convert_cv2qt(self.frame)
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            #     xmin    ymin     xmax   ymax  confidence  class    name
            for line in self.rectPos:
                print(line)
                rect = QRect( int(line[0]), int(line[1]), int(line[2]-line[0]), int(line[3]-line[1]) )
                painter.drawRect(rect)
                painter.drawText(QPoint(int(line[0]),int(line[1])),"{:.3f}:{}".format(line[4],NAME_LIST[int(line[5])]) )
            self.setPixmap(pixmap)
            self.NewFrameReady = False
        else:
            pass

class SimpleAsFuckGui(QMainWindow):
    def __init__(self, parent=None):
        super(SimpleAsFuckGui, self).__init__(parent)
        self.showLbl = ShowResLabel(self)
        self.setCentralWidget(self.showLbl)
# Images

#img2 = cv2.imread('bus.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)
#imgs = [img1, img2]  # batch of images


#from top left
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
if __name__ == '__main__':
    app = QApplication(sys.argv)
    hello_world_gui = SimpleAsFuckGui()
    hello_world_gui.setGeometry(100, 100, 1500, 900)
    hello_world_gui.showLbl.setGeometry(0, 0, 1000, 800)
    hello_world_gui.show()
    ######################################
    #hello_world_gui.showLbl.SetFrame(dummy_res,dummy_img)
    while True:
        chk, frame = VC.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not chk:
            print("not read")
            continue
        res = model([frame], size=640).xyxy[0]

        hello_world_gui.showLbl.SetFrame(res, frame)
        hello_world_gui.showLbl.repaint()
        input()


    sys.exit(app.exec_())