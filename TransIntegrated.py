import cv2
import numpy as np
from TranMarker import MarkerTracker
from Trancontour import CameraProcessor

import cv2
import numpy as np

class IntegratedSystem:
    def __init__(self, camera_index=0):
        self.camera_processor = CameraProcessor(camera_index)
        self.marker_tracker = MarkerTracker(camera_index)
        self.camera_processor.init_camera()  # 确保CameraProcessor完成初始化

    def run(self):
        while True:
            ret, frame = self.camera_processor.cap.read()
            if not ret:
                break
            
            # 处理帧以进行轮廓检测
            frame_for_contours = self.camera_processor.init(frame.copy())
            mask = self.camera_processor.process_frame(frame_for_contours, self.camera_processor.avg_frame)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 200:
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)

            # 处理帧以进行标记跟踪
            frame_for_tracking = self.marker_tracker.init(frame.copy())
            if self.marker_tracker.frame_count < 10:
                mask = self.marker_tracker.find_marker(frame_for_tracking)
                centers = self.marker_tracker.marker_center(mask, frame_for_tracking)
                self.marker_tracker.initial_frame_buffer.append(centers)
                self.marker_tracker.frame_count += 1
                if self.marker_tracker.frame_count == 10:
                    self.marker_tracker.tracked_points, self.marker_tracker.initial_points = self.marker_tracker.initialize_filters(self.marker_tracker.initial_frame_buffer)
            else:
                frame = self.marker_tracker.process_frame(frame_for_tracking)

            cv2.imshow('Integrated Vision Output', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.camera_processor.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = IntegratedSystem()
    system.run()
