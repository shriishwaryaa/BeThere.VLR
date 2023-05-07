# Replacing a Person in a Video with Another

Our project involves developing a series of models capable of replacing a target character in a video with an ego character. The complete pipeline of our project is visually represented in the figure below. The initial step of our project involves processing a sinlge image of the individual to be inserted into the video using Mediapipe pose [1] to obtain their pose. Each frame of the target video is then analyzed by Mediapipe pose to extract the target's pose, enabling us to segment them from the video's background. Next, the holes in the background are filled using the Copy-and-Paste network [2]. Finally, the individual's image is transformed to match the target's pose using a Progressive Pose Attention Transfer network [3] and re-inserted into the filled video, effectively replacing the target with the inserted individual. This methodology enhances the overall realism and believability of the resulting video.

![Screenshot from 2023-05-07 16-53-00](https://user-images.githubusercontent.com/46493008/236704286-b6472c68-10fa-4a1c-9ce6-0601f3d965a6.png)

## Individual Results

### Pose Detection using Mediapipe
![Screenshot from 2023-05-07 17-55-28](https://user-images.githubusercontent.com/46493008/236704398-0451c0ce-2380-494d-a9f7-6015a6dc67a8.png)

### Video Inpainting Using Copy-and-Paste Network

#### Original video
https://user-images.githubusercontent.com/46493008/236704520-892a390c-2bd8-4893-92ed-a6000beb39a1.mp4

#### Video with filled background
https://user-images.githubusercontent.com/46493008/236704590-67c8877d-a94a-4993-b960-61284726a1aa.mp4

### Progressive Pose Attention Transfer network
![Screenshot from 2023-05-07 18-03-11](https://user-images.githubusercontent.com/46493008/236704671-5b84d5bf-ca91-4855-8b74-1d0cbba6afde.png)

#### Videos
https://user-images.githubusercontent.com/46493008/236704801-c31ee454-4d41-4b5e-85d2-2b0635d3d770.mp4

https://user-images.githubusercontent.com/46493008/236704804-1973dad0-db01-4487-b6dd-4d55fc7190b4.mp4

## Final Videos
https://user-images.githubusercontent.com/46493008/236704901-9a2b5ee2-0c9b-4029-a3ea-fc2fbe139f99.mp4

https://user-images.githubusercontent.com/46493008/236704907-3d123cf3-9b75-42d1-9473-d6566fe8a18c.mp4

## References
[1] https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md

[2] Lee, Sungho, et al. "Copy-and-paste networks for deep video inpainting." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.

[3] Zhen Zhu1, Tengteng Huang, Baoguang Shi, Miao Yu, Bofei Wang, Xiang Bai. Progressive Pose Attention Transfer for Person Image Generation. In CVPR, 2019





