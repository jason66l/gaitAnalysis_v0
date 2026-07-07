Prototyped a mobile gait analysis application, using only an iPhone and
consumer hardware.
Initially, experimented with pretrained pose-detection models such as OpenPose. However, due
to inconsistent accuracy in real-world lighting and occlusion conditions, pivoted to a more
controlled approach using colored motion markers and OpenCV-based image tracking.
We implemented a custom tracking pipeline that:

*Detects color markers placed on the leg
*Calculates joint angles over time (coronal and sagittal planes)
*Measures cadence and stride length from positional tracking

Later moved Python code to XCode to utilize CoreMotion
*only Python prototype was attached due to NDA--full demo shown below, however


https://github.com/user-attachments/assets/65492b09-b08a-4282-969d-b829ccbcb4aa


