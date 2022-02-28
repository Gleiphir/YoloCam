import cv2
import threading

class VCThread(threading.Thread):
    def __init__(self,vc_id):
        super(VCThread, self).__init__()
        self.VC = cv2.VideoCapture(vc_id)
        #self.lock = threading.Lock()
        self.frame =None

    def getFrame(self):
        return self.frame


    def run(self) -> None:
        while True:
            if not self.VC.isOpened():
                raise IOError("VideoCapture not set")
            rec, frame = self.VC.read()
            if not rec:
                continue
            else:
                self.frame = frame

