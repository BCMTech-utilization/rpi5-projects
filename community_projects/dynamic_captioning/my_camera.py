import cv2

class Camera :
    def __init__(self, video) :
        # 비디오캡처 객체 생성
        self.cap= cv2.VideoCapture(video)
        self.video= video
        if not self.cap.isOpened() :
            cap.release()
            raise RuntimeError("카메라를 사용할 수 없습니다.")

        # 비디오 사이즈 명시 및 정의
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 352)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


    def get_picture(self) :
        return self.cap.read() # bool, img_array

    
    def show_picture(self, pic) :
        cv2.imshow("caption.py", pic)
        cv2.waitKey(1)

    def reset(self) :
        self.cap= cv2.VideoCapture(self.video)
