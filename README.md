# Yolov7-ResNet-Fire-Detection

- ip 주소, ROS_URI, ROS_IP는 현재 내 주소로 변경
- nano ~/.bashrc -> ROS_MASTER_URI & ROS_IP는 현재 와이파이 IPv4로 변경
``` roscore  ```
- roslaunch 명령을 통해 런치 파일 실행 -> yolov7_object_detection.py 스크립트 실행하는 노드 시작
``` cd yolov7_ws ```
``` roslaunch yolov7 yolov7_object_detection.launch ```
- rosbag 실행
``` rosbag play near.bag -l ```
- rosrun 실행 (rosbag info로 토픽 이름 확인 -> 이름 변경)
``` rosrun image_view image_view /image:=/jackal/image_bbox ```

### pt_detect.py 코드
- fire 감지하면, "/emergency_call" Bool 메시지 publish 하도록 함
- yolo 이미지 검출을 1초에 1~2번만 하도록 함
- compressed image로 처리
- 코드 실행을 shell script로 구현
``` sudo nano fire_detection.sh ```
<img width="232" alt="image" src="https://github.com/chaewonS/Yolov7-ResNet-Fire-Detection/assets/81732426/8849fbb1-f28f-4728-99bc-385e10971b25">
``` chmod +x fire_detection.sh ```
``` ./fire_detection.sh ```
