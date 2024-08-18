import cv2
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

class MarkerTrackerBlack:
    def __init__(self, camera_index=0):
        
        self.cap = cv2.VideoCapture(camera_index)
        self.tracked_points = []
        self.initial_points = []
        self.initial_frame_buffer = []
        self.frame_count = 0
        self.RESCALE = 1  # 假设RESCALE值为1
        

    def init(self, frame):
        return cv2.resize(frame, (0, 0), fx=1.0 / self.RESCALE, fy=1.0 / self.RESCALE)

    def reduce_noise(self, frame):
        blurred = cv2.GaussianBlur(frame, (31, 31), 2)
        return blurred

    def enhance_contrast(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(100, 100))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced_image

    def find_marker(self, frame):
        noise_reduced_frame = self.reduce_noise(frame)
        enhanced_frame = self.enhance_contrast(noise_reduced_frame)
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 201, -100)
        kernel_open = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        kernel_close = np.ones((15, 15), np.uint8)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_close)
        return morph

    def marker_center(self, mask, frame):
        areaThresh1 = 10
        areaThresh2 = 1200
        MarkerCenter = []

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            AreaCount = cv2.contourArea(contour)
            if areaThresh1 < AreaCount < areaThresh2 and abs(np.max([w, h]) / np.min([w, h]) - 1) < 5:
                M = cv2.moments(contour)
                mc = [M['m10'] / M['m00'], M['m01'] / M['m00']]
                MarkerCenter.append(mc)
                cv2.circle(frame, (int(mc[0]), int(mc[1])), 1, (0, 0, 255), 2, 6)
  
        return MarkerCenter

    def initialize_filters(self, frames):
        point_counts = defaultdict(int)
        all_centers = [frame for frame in frames if frame]

        for i in range(len(all_centers)):
            for j in range(i + 1, len(all_centers)):
                if all_centers[i] and all_centers[j]:
                    if len(all_centers[i]) > 0 and len(all_centers[j]) > 0:
                        distances = distance.cdist(all_centers[i], all_centers[j], 'euclidean')
                        row_ind, col_ind = linear_sum_assignment(distances)
                        for row, col in zip(row_ind, col_ind):
                            if distances[row, col] < 50:
                                point_counts[tuple(all_centers[i][row])] += 1
                                point_counts[tuple(all_centers[j][col])] += 1

        initial_points = [point for point, count in point_counts.items() if count >= 3]
        tracked_points = []

        for point in initial_points:
            tracked_points.append(point)

        return tracked_points, initial_points

    def process_frame(self, frame):
        mask = self.find_marker(frame)
        centers = self.marker_center(mask, frame)
        centers = [tuple(center) for center in centers]

        if centers:
            distances = distance.cdist(self.tracked_points, centers, 'euclidean')
            row_ind, col_ind = linear_sum_assignment(distances)

            new_tracked_points = [None] * len(self.tracked_points)

            for row, col in zip(row_ind, col_ind):
                if distances[row, col] < 20:
                    new_tracked_points[row] = centers[col]
                    current_point = centers[col]
                    initial_point = self.initial_points[row]
                    line_length = np.linalg.norm(np.array(initial_point) - np.array(current_point))
                    cv2.circle(frame, (int(initial_point[0]), int(initial_point[1])), 5, (0, 255, 0), -1)
                    cv2.circle(frame, (int(current_point[0]), int(current_point[1])), 5, (255, 0, 0), -1)
                    cv2.line(frame, (int(initial_point[0]), int(initial_point[1])), (int(current_point[0]), int(current_point[1])), (0, 0, 0), 2)
                                    # 如果连线长度大于阈值，圈出这个点
                    if line_length > 20:
                        cv2.circle(frame, (int(current_point[0]), int(current_point[1])), 10, (0, 0, 255), 2)

                    
            self.tracked_points = [p if p is not None else self.tracked_points[i] for i, p in enumerate(new_tracked_points)]

        return frame

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = self.init(frame)

            if self.frame_count < 10:
                mask = self.find_marker(frame)
                centers = self.marker_center(mask, frame)
                centers = [tuple(center) for center in centers]
                self.initial_frame_buffer.append(centers)
                self.frame_count += 1

                if self.frame_count == 10:
                    self.tracked_points, self.initial_points = self.initialize_filters(self.initial_frame_buffer)
            else:
                frame = self.process_frame(frame)

            cv2.imshow('Frame with Black Points', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = MarkerTrackerBlack()
    tracker.run()
