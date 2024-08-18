import cv2
import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

class MarkerTracker:
    def __init__(self, camera_index=1):
        
        self.cap = cv2.VideoCapture(camera_index)
        self.tracked_points = []
        self.initial_points = []
        self.initial_frame_buffer = []
        self.frame_count = 0
        self.RESCALE = 1  # 假设RESCALE值为1
        self.init_camera()

    def init_camera(self):
        # Set up the window for trackbars
        cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Trackbars', 600, 500)

        # Create trackbars
        cv2.createTrackbar('Clip Limit', 'Trackbars', 2, 10, self.nothing)
        cv2.createTrackbar('Tile Grid Size', 'Trackbars', 8, 32, self.nothing)
        cv2.createTrackbar('Gaussian Kernel Size', 'Trackbars', 5, 15, self.nothing)
        cv2.createTrackbar('Threshold Type', 'Trackbars', 0, 3, self.nothing)
        cv2.createTrackbar('Binary Threshold', 'Trackbars', 128, 255, self.nothing)
        cv2.createTrackbar('Otsu Adjustment', 'Trackbars', 10, 20, self.nothing)
        cv2.createTrackbar('Adaptive Method', 'Trackbars', 0, 1, self.nothing)
        cv2.createTrackbar('Adaptive Block Size', 'Trackbars', 11, 25, self.nothing)
        cv2.createTrackbar('Adaptive C', 'Trackbars', 0, 20, self.nothing)
        cv2.createTrackbar('Morph Open Size', 'Trackbars', 3, 15, self.nothing)
        cv2.createTrackbar('Morph Close Size', 'Trackbars', 15, 50, self.nothing)   

    def nothing(self, x):
        pass
    def init(self, frame):

        return cv2.resize(frame, (0, 0), fx=1.0 / self.RESCALE, fy=1.0 / self.RESCALE)


    def enhance_contrast(self, frame):
        """
        增强图像对比度，同时增加蓝色光的强度。
        参数:
            frame: 原始图像帧
        返回:
            对比度增强且蓝色光增强后的图像帧
        """
        # 增强图像对比度
        clip_limit = cv2.getTrackbarPos('Clip Limit', 'Trackbars')
        tile_grid_size = cv2.getTrackbarPos('Tile Grid Size', 'Trackbars')
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        # clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(10, 10))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 增加蓝色光的强度
        enhanced_image[:, :, 0] = np.clip(enhanced_image[:, :, 0].astype(np.float32) * 0.1, 0, 255).astype(np.uint8)
        
        return enhanced_image

    def find_marker(self, frame):
        kernel_size = cv2.getTrackbarPos('Gaussian Kernel Size', 'Trackbars')
        kernel_size = max(1, kernel_size)  # Ensure kernel size is at least 1
        if kernel_size % 2 == 0:  # Make kernel size odd if it is even
            kernel_size += 1
        gaussian_blur_value = cv2.getTrackbarPos('Threshold Type', 'Trackbars')  # This might be incorrect usage. Check if this is what you intended.
        # Apply Gaussian blur with the corrected kernel size

        enhanced_frame = self.enhance_contrast(frame)
        cv2.imshow('enhanced_frame', enhanced_frame)

        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)

        noise_reduced_frame = cv2.GaussianBlur(gray, (15, 15), 0)
        noise_reduced_frame = cv2.GaussianBlur(gray, (kernel_size, kernel_size), gaussian_blur_value)
        cv2.imshow('noise_reduced_frame', noise_reduced_frame)

        # thresh = cv2.adaptiveThreshold(noise_reduced_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 40)
        # # cv2.imshow('thresh', thresh)
        # kernel_open = np.ones((12, 12), np.uint8)
        # morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

        # # cv2.imshow('morph1', morph)
        # kernel_close = np.ones((32, 32), np.uint8)
        # morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_close)
        # # cv2.imshow('morph', morph)
        thresh_type = cv2.getTrackbarPos('Threshold Type', 'Trackbars')
        if thresh_type == 0:  # Binary
            binary_thresh = cv2.getTrackbarPos('Binary Threshold', 'Trackbars')
            _, thresh = cv2.threshold(noise_reduced_frame, binary_thresh, 255, cv2.THRESH_BINARY)
        elif thresh_type == 1:  # Binary INV
            binary_thresh = cv2.getTrackbarPos('Binary Threshold', 'Trackbars')
            _, thresh = cv2.threshold(noise_reduced_frame, binary_thresh, 255, cv2.THRESH_BINARY_INV)
        elif thresh_type == 2:  # Adaptive
            adaptive_method = cv2.getTrackbarPos('Adaptive Method', 'Trackbars')
            adaptive_block_size = cv2.getTrackbarPos('Adaptive Block Size', 'Trackbars')
            adaptive_c = cv2.getTrackbarPos('Adaptive C', 'Trackbars')
            method = cv2.ADAPTIVE_THRESH_MEAN_C if adaptive_method == 0 else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            thresh = cv2.adaptiveThreshold(noise_reduced_frame, 255, method, cv2.THRESH_BINARY, adaptive_block_size, adaptive_c)
        elif thresh_type == 3:  # Otsu
            _, thresh = cv2.threshold(noise_reduced_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        morph_open_size = cv2.getTrackbarPos('Morph Open Size', 'Trackbars')
        morph_close_size = cv2.getTrackbarPos('Morph Close Size', 'Trackbars')
        kernel_open = np.ones((morph_open_size, morph_open_size), np.uint8)
        kernel_close = np.ones((morph_close_size, morph_close_size), np.uint8)
        cv2.imshow('thresh', thresh)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        cv2.imshow('morph', morph)
        # morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_close)



        return morph

    def marker_center(self, mask, frame):
        areaThresh1 = 50
        areaThresh2 = 3000
        MarkerCenter = []

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            AreaCount = cv2.contourArea(contour)
            # if areaThresh1 < AreaCount < areaThresh2 and abs(np.max([w, h]) / np.min([w, h]) - 1) < 5:
            if areaThresh1 < AreaCount < areaThresh2 and abs(np.max([w, h]) / np.min([w, h]) - 1) < 2:
                M = cv2.moments(contour)
                mc = [M['m10'] / M['m00'], M['m01'] / M['m00']]
                MarkerCenter.append(mc)
                cv2.circle(frame, (int(mc[0]), int(mc[1])), 1, (0, 0, 255), 2, 6)

        return MarkerCenter

    def initialize_filters(self, frames):
        point_counts = defaultdict(int)
        print(point_counts)
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

        if not centers or not self.tracked_points:
            # No centers or no tracked points, skip processing
            return frame

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
                if line_length > 20:
                    cv2.circle(frame, (int(current_point[0]), int(current_point[1])), 10, (0, 0, 255), 2)

        self.tracked_points = [p if p is not None else self.tracked_points[i] for i, p in enumerate(new_tracked_points)]

        return frame

    # def run(self):
    #     while True:
    #         ret, frame = self.cap.read()
    #         if not ret:
    #             break

    #         frame = self.init(frame)

    #         if self.frame_count < 10:
    #             mask = self.find_marker(frame)
    #             centers = self.marker_center(mask, frame)
    #             centers = [tuple(center) for center in centers]
    #             self.initial_frame_buffer.append(centers)
    #             self.frame_count += 1

    #             if self.frame_count == 10:
    #                 self.tracked_points, self.initial_points = self.initialize_filters(self.initial_frame_buffer)
    #         else:
    #             frame = self.process_frame(frame)

    #         cv2.imshow('Frame with Black Points', frame)

    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord('q'):
    #             break

    #     self.cap.release()
    #     cv2.destroyAllWindows()
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (0, 0), fx=1.0 / self.RESCALE, fy=1.0 / self.RESCALE)
            mask = self.find_marker(frame)
            self.marker_center(mask, frame)
            cv2.imshow('Frame with Markers', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = MarkerTracker()
    tracker.run()
