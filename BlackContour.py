import cv2
import numpy as np

class CameraProcessorBlack:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.avg_frame = None
        self.init_camera()
        
    def init_camera(self):
        # 跳过前100帧以让摄像头预热，避免初始化时的暗影问题
        for _ in range(50):
            ret, _ = self.cap.read()
            if not ret:
                print("Failed to skip frame during camera warm-up")
                break
        self.collect_reference_frame()
    
    def collect_reference_frame(self):
        count = 0
        avg_frame = None
        
        # 收集前20帧并计算平均
        while count < 20:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = self.init(frame)
            if avg_frame is None:
                avg_frame = np.float32(frame)
            else:
                cv2.accumulate(frame, avg_frame)
            count += 1
        
        # 计算平均参考帧
        self.avg_frame = cv2.convertScaleAbs(avg_frame / 20)
    
    def init(self, frame):
        RESCALE = 1  # 保持原始分辨率
        return cv2.resize(frame, (0, 0), fx=1.0 / RESCALE, fy=1.0 / RESCALE)
    
    def reduce_noise(self, frame):
        # 使用高斯模糊减少噪点
        # blurred = cv2.GaussianBlur(frame, (101, 101), 20)
        blurred = cv2.GaussianBlur(frame, (45, 45), 0)
        return blurred
    
    def enhance_contrast(self, frame):
        """
        增强图像对比度，同时增加蓝色光的强度。
        参数:
            frame: 原始图像帧
        返回:
            对比度增强且蓝色光增强后的图像帧
        """
        # 增强图像对比度
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(100, 100))
        clahe = cv2.createCLAHE(clipLimit=13, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 增加蓝色光的强度
        enhanced_image[:, :, 0] = np.clip(enhanced_image[:, :, 0].astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
        
        return enhanced_image
    
    def process_frame(self, frame, reference_frame):
        # 帧与参考帧相减
        diff = cv2.absdiff(frame, reference_frame)
        # cv2.imshow('diff', diff)
        
        # 减少噪点和增强对比度
        noise_reduced_frame = self.reduce_noise(diff)
        # cv2.imshow('noise_reduced_frame', noise_reduced_frame)
        
        # 转换为灰度图像并使用阈值检测轮廓
        gray = cv2.cvtColor(noise_reduced_frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', gray)
        
        # _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 1001, -9)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if _ < 0:
            adjusted_threshold = _ - 13 
        else:
            adjusted_threshold = _ + 13  
        _, thresh = cv2.threshold(gray, adjusted_threshold, 255, cv2.THRESH_BINARY)  # Apply the adjusted threshold




        # cv2.imshow('thresh', thresh)
        kernel = np.ones((6, 6), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # 形态学操作以清晰轮廓
        kernel = np.ones((60, 60), np.uint8)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('morph', morph)
        
        return morph
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self.init(frame)
            mask = self.process_frame(frame, self.avg_frame)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
            cv2.imshow('Contour Detection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = CameraProcessorBlack()
    processor.run()
