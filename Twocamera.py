import cv2
import numpy as np
from TranMarker import MarkerTracker
from TranCounter import CameraProcessor
from BlackMarker import MarkerTrackerBlack
from BlackCounter import CameraProcessorBlack

class DualIntegratedSystem:
    def __init__(self, camera_index1=0, camera_index2=1):
        # 第一个系统
        self.camera_processor1 = CameraProcessor(camera_index1)
        self.marker_tracker1 = MarkerTracker(camera_index1)
        
        # 第二个系统
        self.camera_processor2 = CameraProcessorBlack(camera_index2)
        self.marker_tracker2 = MarkerTrackerBlack(camera_index2)

        # 初始化两个摄像头
        self.camera_processor1.init_camera()
        self.camera_processor2.init_camera()

    def run(self):
        while True:
            # 处理第一个摄像头
            ret1, frame1 = self.camera_processor1.cap.read()
            if ret1:
                frame1 = self.process_frame(self.camera_processor1, self.marker_tracker1, frame1)
            frame1 = cv2.flip(frame1, 0) 
            # 处理第二个摄像头
            ret2, frame2 = self.camera_processor2.cap.read()
            if ret2:
                frame2 = self.process_frame(self.camera_processor2, self.marker_tracker2, frame2)

            # 显示结果
            if ret1 and ret2:
                combined_frame = np.hstack((frame1, frame2))  # 水平堆叠两个帧
                cv2.imshow('Dual Integrated Vision Output', combined_frame)
            elif ret1:
                cv2.imshow('Camera 1 Output', frame1)
            elif ret2:
                cv2.imshow('Camera 2 Output', frame2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.camera_processor1.cap.release()
        self.camera_processor2.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, processor, tracker, frame):
        # frame_for_contours = processor.init(frame.copy())
        # mask = processor.process_frame(frame_for_contours, processor.avg_frame)
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for contour in contours:
        #     if cv2.contourArea(contour) > 200:
        #         cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)

        frame_for_tracking = tracker.init(frame.copy())
        if tracker.frame_count < 10:
            mask = tracker.find_marker(frame_for_tracking)
            centers = tracker.marker_center(mask, frame_for_tracking)
            tracker.initial_frame_buffer.append(centers)
            tracker.frame_count += 1
            if tracker.frame_count == 10:
                tracker.tracked_points, tracker.initial_points = tracker.initialize_filters(tracker.initial_frame_buffer)
        else:
            frame = tracker.process_frame(frame_for_tracking)
        return frame

if __name__ == "__main__":
    system = DualIntegratedSystem()
    system.run()
