# MvAIE-for-Image-Enhancement
# Multi-view Adaptive Image Enhancement with Hierarchical Attention for Complex Underground Mining Scenes
The Multi-view Adaptive Image Enhancement with Hierarchical Attention Network (MvAIE) is designed to enhance images in underground mines, specifically tackling challenges posed by darkness, fog, and glare. 

This code was collaboratively developed by Wuhan University of Technology and The Hong Kong Polytechnic University to enhance perception capabilities for autonomous driving in underground mines, facilitating the practical implementation of autonomous driving technology in such environments.

# Abstract
The harsh environmental conditions in underground mines, including low illumination, haze, and dust, pose significant challenges to the visual perception systems of autonomous mining vehicles. To address these challenges, this paper proposes a novel multi-view adaptive image enhancement with Hierarchical Attention (MvAIE) that combines adaptive enhancement with multi-view feature fusion strategies to achieve efficient and precise image restoration. Specifically, the proposed adaptive enhancement module dynamically adjusts gamma correction and HSV color space parameters in response to varying lighting and color conditions. Furthermore, the multi-branch feature fusion module leverages dilated convolutions and a hierarchical encoder-decoder structure to extract features across multiple scales, capturing both global context and fine details. It is particularly effective for recovering essential features such as object edges and textures in degraded images, directly benefiting task-specific performance. To further enhance task relevance, the framework incorporates a multi-receptive global-local attention mechanism that prioritizes critical image regions, improving the network’s focus on areas important for accurate object detection and localization. By aligning the enhancement process with the requirements of downstream tasks, MvAIE ensures that restored images not only achieve superior visual quality but also provide robust support for autonomous mining applications. Experimental results demonstrate that MvAIE outperforms existing methods in enhancing image quality, providing robust support for autonomous driving in intelligent mining applications.


![Figure_2](https://github.com/user-attachments/assets/f4df3958-ef1b-4a89-99bb-c0bbda40cee6)


