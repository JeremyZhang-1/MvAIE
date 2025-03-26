# MvAIE-for-Image-Enhancement
# Multi-view Adaptive Image Enhancement with Hierarchical Attention for Complex Underground Mining Scenes
#  Introduction
The Multi-view Adaptive Image Enhancement with Hierarchical Attention Network (MvAIE) is designed to enhance images in underground mines, specifically tackling challenges posed by darkness, fog, and glare. 

This code was collaboratively developed by Wuhan University of Technology and The Hong Kong Polytechnic University to enhance perception capabilities for autonomous driving in underground mines, facilitating the practical implementation of autonomous driving technology in such environments.

# Abstract
The harsh environmental conditions in underground mines, including low illumination, haze, and dust, pose significant challenges to the visual perception systems of autonomous mining vehicles. To address these challenges, this paper proposes a novel multi-view adaptive image enhancement with Hierarchical Attention (MvAIE) that combines adaptive enhancement with multi-view feature fusion strategies to achieve efficient and precise image restoration. Specifically, the proposed adaptive enhancement module dynamically adjusts gamma correction and HSV color space parameters in response to varying lighting and color conditions. Furthermore, the multi-branch feature fusion module leverages dilated convolutions and a hierarchical encoder-decoder structure to extract features across multiple scales, capturing both global context and fine details. It is particularly effective for recovering essential features such as object edges and textures in degraded images, directly benefiting task-specific performance. To further enhance task relevance, the framework incorporates a multi-receptive global-local attention mechanism that prioritizes critical image regions, improving the network’s focus on areas important for accurate object detection and localization. By aligning the enhancement process with the requirements of downstream tasks, MvAIE ensures that restored images not only achieve superior visual quality but also provide robust support for autonomous mining applications. Experimental results demonstrate that MvAIE outperforms existing methods in enhancing image quality, providing robust support for autonomous driving in intelligent mining applications.

# Introduction
Coal remains a vital element of the global energy landscape, with underground mining areas at the heart of its production process. Underground mining environments present complex operational challenges due to inadequate lighting along with haze and dust which create dangerous terrain conditions. Manual mining operations become much more hazardous and challenging due to these working conditions. Autonomous mining vehicles (AMV) reduce human exposure to dangerous zones while lowering accident risks and facilitate round-the-clock operations. The implementation of autonomous systems in underground mining faces significant imaging challenges. Experts suggest implementing low-visibility image enhancement techniques to help AMV perform precise environmental perception even in complex conditions which will lead to safer and more efficient operations. Image enhancement methods are divided into two main categories which include physical-based and learning-based methods.

![Figure07](https://github.com/user-attachments/assets/e121bfff-6ede-4481-890b-40fc8dbc4c37)

# Mine imaging degradation
![Figure_3_1](https://github.com/user-attachments/assets/055a3984-6277-4fac-a29f-3ce1cd92ed0b)

# Multi-view adaptive image enhancement network
![Figure_2_1](https://github.com/user-attachments/assets/4e3ce85e-67be-481c-bbf8-3ea811a0a555)

# Prerequisites
```
conda create -n dehaze python=3.7
conda activate dehaze
conda install pytorch=1.10.2 torchvision torchaudio cudatoolkit=11.3 -c pytorch
python3 -m pip install scipy==1.7.3
python3 -m pip install opencv-python==4.4.0.46
```

# Result

The visual comparison result on (a) synthesized degraded CDD-11 images with (b) 4kdehaze , (c) APSF , (d) C2PNet, (e) DHFormer, (f) GMLC, (g) IENHC, (h) MRP, (i) OSFD, (j) Our method, and (k) Gronud Truth, respectively.
![Figure15](https://github.com/user-attachments/assets/4bb2c1cf-9e06-404e-a000-ed0368e1cb2c)

The visual comparison result on (a) synthesized degraded mining images with (b) 4kdehaze , (c) APSF , (d) C2PNet, (e) DHFormer, (f) GMLC, (g) IENHC, (h) MRP, (i) OSFD, (j) Our method, and (k) Gronud Truth, respectively.
![Figure16](https://github.com/user-attachments/assets/5170c1c4-fd96-468b-9fe9-d57f2f92c7cf)

# License
The code and models in this repository are licensed under the WHUT License for academic and other non-commercial uses.
For commercial use of the code and models, separate commercial licensing is available. Please contact:
- Jingming Zhang (jeremy.zhang@whut.edu.cn)
- Yuxu LU (yuxulouis.lu@connect.polyu.hk)

