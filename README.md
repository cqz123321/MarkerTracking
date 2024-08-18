# MarkerTracking
For DIgit transparent and black skin 
BlackCoutour.py, BlackMArker.py, and Blackintegrated.py are used together for black skin DIgit tactile sensor's Marker tracking and Contour Detection.
TranCoutour.py, TranMArker.py, and Tranintegrated.py are used together for Transparent skin Digit tactile sensor's Marker tracking and Contour Detection.
TranMArkerBarAdjust is used for the DIgit transparent skin to manually adjust the setting bar, to find the optimal value for finding the marker.

MarkerTracker的简要概括

enhance_contrast 方法:
增强输入帧的对比度，并增加蓝色光的强度（因为digit的蓝光较弱），以便更容易检测到标记。

find_marker 方法:
提升对比度后将帧转换为灰度图像。
通过高斯模糊和自适应阈值方法减少噪声并生成二值化图像。
通过形态学操作清除噪声，返回处理后的二值掩码。

marker_center 方法:
从处理后的二值掩码中找到轮廓，计算其中心坐标。
使用面积和宽高比的阈值来，寻找比较符合标准的标记点，过滤噪音，并将其中心点标记在原始帧中。

initialize_filters 方法:
初始化跟踪器，计算每个初始帧中标记点之间的欧式距离，50为界限。
因为初始化的点可能有移动，并不呆在原地， 
所以通过分配算法找到10帧，每帧移动距离不超过X，并稳定存在（十个里面超过六个）点作为初始跟踪点。

process_frame 方法:

使用 distance.cdist 计算上一帧的跟踪点与当前帧中的标记点之间的欧氏距离。
使用 linear_sum_assignment 找到跟踪点和当前标记点之间的最佳匹配，这个函数找到跟踪点和中心点之间使总距离最小）
如果当前跟踪点与新找到的标记点之间的距离大于一定阈值，则用红圈标出跟踪点。
在帧中绘制当前跟踪点与初始点之间的连线。
最后，self.tracked_points 被更新为新的跟踪点位置，如果某些点没有匹配到（new_tracked_points 中为 None），就保持它们在上一帧中的位置不变。


run 方法:

持续从摄像头读取帧并处理。
前10帧用于初始化跟踪器（通过 initialize_filters 方法）。
在初始化完成后，处理后续帧并显示结果。
用户可以按下 q 键退出循环并结束程序。


