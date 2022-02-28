import sys
import time

import torch
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2

CAM_API = cv2.CAP_MSMF
############
# model
print("Model Init")
torch.hub.set_dir("./hub")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').cuda()

dummy_img = cv2.imread('bus.jpg')
dummy_img = cv2.cvtColor(dummy_img, cv2.COLOR_BGR2RGB)

dummy_batch = [dummy_img]

results = model(dummy_batch, size=640)

NAME_LIST = results.pandas().names
#####################################
def convert_cv2qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    #rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    #p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
    return QPixmap.fromImage(convert_to_Qt_format)

def unPackRes(res:torch.FloatTensor):
    return res.cpu().numpy().tolist()



class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel(self)

        self.VBL.addWidget(self.FeedLabel,alignment=Qt.AlignTop)

        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN,alignment=Qt.AlignTop)

        self.Worker1 = Worker1()

        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)

        self.setLayout(self.VBL)
        self.Worker1.start()

    @pyqtSlot(QPixmap)
    def ImageUpdateSlot(self, pixmap):
        self.FeedLabel.setPixmap(pixmap)
        #self.FeedLabel.repaint()

    def CancelFeed(self):
        self.Worker1.stop()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QPixmap)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0,CAM_API)
        while self.ThreadActive:
            if not Capture or not Capture.isOpened():
                Capture = cv2.VideoCapture(0, CAM_API)
            ret, frame = Capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = unPackRes(model(Image, size=640).xyxy[0])
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                #
                pixmap = QPixmap.fromImage(Pic)
                painter = QPainter(pixmap)
                painter.setPen(QPen(Qt.red, 2, Qt.DotLine))
                #     xmin    ymin     xmax   ymax  confidence  class    name
                for line in res:
                    print(line)
                    rect = QRect(int(line[0]), int(line[1]), int(line[2] - line[0]), int(line[3] - line[1]))
                    painter.drawRect(rect)
                    painter.drawText(QPoint(int(line[0]), int(line[1])),
                                     "{:.3f}:{}".format(line[4], NAME_LIST[int(line[5])]))

                self.ImageUpdate.emit(pixmap)
                print("-------")
                #time.sleep(0.1)
                self.msleep(500)
    def stop(self):
        self.ThreadActive = False
        self.quit()

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())