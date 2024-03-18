# awesome-Stereo
A curated list of Stereo Matching resources

## Table of Contents

- [Survey](#survey)
- [Papers](#papers)
- [Talks](#talks)
- [Implementations](#implementations)

## Survey

## Datasets

#### Real-World - Passive

* **Middlebury 2021 Mobile Dataset**: [Website](https://vision.middlebury.edu/stereo/data/scenes2021/)

* **The Booster Dataset**: Open Challenges in Deep Stereo: The Booster Dataset, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ramirez_Open_Challenges_in_Deep_Stereo_The_Booster_Dataset_CVPR_2022_paper.pdf)] [[Website](https://cvlab-unibo.github.io/booster-web/)]

* **Holopix50k**: Holopix50k: A Large-Scale In-the-Wild Stereo Image Dataset, *CVPR, 2020*. [[Website](https://leiainc.github.io/holopix50k/)] [[Paper](https://arxiv.org/abs/2003.11172)]

* **InStereo2K**: InStereo2K: A Large Real Dataset for Stereo Matching in Indoor Scenes, *Science China Information Sciences, 2020*. [[Paper](https://link.springer.com/article/10.1007/s11432-019-2803-x)] [[Code](https://github.com/YuhuaXu/StereoDataset)]


#### Real-World - Multimodal

* **DSEC**: DSEC: A Stereo Event Camera Dataset for Driving Scenarios, *RAL, 2021*. [[Paper](https://rpg.ifi.uzh.ch/docs/RAL21_DSEC.pdf)] [[Code](https://github.com/uzh-rpg/DSEC)] [[Website](https://dsec.ifi.uzh.ch/)]

* **Gated Stereo**: Gated Stereo: Joint Depth Estimation from Gated and Wide-Baseline Active Stereo Cues, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Walz_Gated_Stereo_Joint_Depth_Estimation_From_Gated_and_Wide-Baseline_Active_CVPR_2023_paper.pdf)] [[Website](https://light.princeton.edu/gatedstereo/)]

* **RGB-MS**: RGB-Multispectral Matching: Dataset, Learning Methodology, Evaluation, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tosi_RGB-Multispectral_Matching_Dataset_Learning_Methodology_Evaluation_CVPR_2022_paper.pdf)] [[Website](https://cvlab-unibo.github.io/rgb-ms-web/)]

* **MS^2**: Deep Depth Estimation From Thermal Image, *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Shin_Deep_Depth_Estimation_From_Thermal_Image_CVPR_2023_paper.pdf)] [[Website](https://sites.google.com/view/multi-spectral-stereo-dataset)]


#### Synthetic

* **HS-VS**: Hierarchical deep stereo matching on high-resolution image, *CVPR, 2019*. [[Paper](https://arxiv.org/pdf/1912.06704.pdf)] [[Code](https://github.com/gengshan-y/high-res-stereo?tab=readme-ov-file)]

* **Tartanair**: Tartanair: A dataset to push the limits of visual slam, *IROS, 2020*. [[Paper](https://ieeexplore.ieee.org/document/9341801?denied=)] [[Website](https://theairlab.org/tartanair-dataset/shorter)]

* **UnrealStereo4K**: SMD-Nets: Stereo Mixture Density Networks, *CVPR, 2021*. [[Paper](http://www.cvlibs.net/publications/Tosi2021CVPR.pdf)] [[Code](https://github.com/fabiotosi92/SMD-Nets)]

* **CREStereo**: Practical stereo matching via cascaded recurrent network with adaptive correlation, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Practical_Stereo_Matching_via_Cascaded_Recurrent_Network_With_Adaptive_Correlation_CVPR_2022_paper.pdf)] [[Code](https://github.com/megvii-research/CREStereo)]

* **SimStereo**: Active-Passive SimStereo – Benchmarking the Cross-Generalization Capabilities of Deep Learning-based Stereo Methods, *NeurIPS, 2022*. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/bc3a68a20e5c8ba5cbefc1ecf74bfaaa-Paper-Datasets_and_Benchmarks.pdf)] [[Website](https://ieee-dataport.org/open-access/active-passive-simstereo)]

* **Spring**: Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Mehl_Spring_A_High-Resolution_High-Detail_Dataset_and_Benchmark_for_Scene_Flow_CVPR_2023_paper.html)] [[Website](http://spring-benchmark.org)]

* **Dynamic Replica**: DynamicStereo: Consistent Dynamic Depth From Stereo Videos, *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Karaev_DynamicStereo_Consistent_Dynamic_Depth_From_Stereo_Videos_CVPR_2023_paper.pdf)] [[Website](https://dynamic-stereo.github.io/)]


## Papers

#### Architectures

* **AANet**: AANet: Adaptive Aggregation Network for Efficient Stereo Matching, *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_AANet_Adaptive_Aggregation_Network_for_Efficient_Stereo_Matching_CVPR_2020_paper.pdf)] [[Code](https://github.com/haofeixu/aanet)]

* **CasStereo**: Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching, *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gu_Cascade_Cost_Volume_for_High-Resolution_Multi-View_Stereo_and_Stereo_Matching_CVPR_2020_paper.pdf)] [[Code](https://github.com/alibaba/cascade-stereo)]

* **Bi3D**: Bi3D: Stereo Depth Estimation via Binary Classifications, *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Badki_Bi3D_Stereo_Depth_Estimation_via_Binary_Classifications_CVPR_2020_paper.pdf)] [[Code](https://github.com/NVlabs/Bi3D)]

* **WaveletStereo**: WaveletStereo: Learning Wavelet Coefficients of Disparity Map in Stereo Matching, *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_WaveletStereo_Learning_Wavelet_Coefficients_of_Disparity_Map_in_Stereo_Matching_CVPR_2020)]

* **AcfNet**: Adaptive Unimodal Cost Volume Filtering for Deep Stereo Matching, *AAAI, 2020*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6991/6845)] [[Code](https://github.com/youmi-zym/AcfNet)]

* **Fadnet**: Fadnet: A Fast and Accurate Network for Disparity Estimation, *ICRA, 2020*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9197031)] [[Code](https://github.com/HKBU-HPML/FADNet)]

* **CDN**: Wasserstein Distances for Stereo Disparity Estimation, *NeurIPS, 2020*. [[Paper](https://papers.nips.cc/paper/2020/file/fe7ecc4de28b2c83c016b5c6c2acd826-Paper.pdf)] [[Code](https://github.com/Div99/W-Stereo-Disp)]

* **SGNet**: SGNet: Semantics Guided Deep Stereo Matching, *ACCV, 2020*. [[Paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Chen_SGNet_Semantics_Guided_Deep_Stereo_Matching_ACCV_2020_paper.pdf)]

* **CFNet**: CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching, *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Shen_CFNet_Cascade_and_Fused_Cost_Volume_for_Robust_Stereo_Matching_CVPR_2021_paper.pdf)] [[Code](https://github.com/gallenszl/CFNet)]

* **HITNet**: HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching, *CVPR, 2021*. [[Paper](https://arxiv.org/abs/2007.12140)] [[Code](https://github.com/XuelianCheng/LEAStereo)]

* **SMD-Nets**: SMD-Nets: Stereo Mixture Density Networks, *CVPR, 2021*. [[Paper](http://www.cvlibs.net/publications/Tosi2021CVPR.pdf)] [[Code](https://github.com/fabiotosi92/SMD-Nets)]

* **BGNet**: Bilateral Grid Learning for Stereo Matching Networks, *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Bilateral_Grid_Learning_for_Stereo_Matching_Networks_CVPR_2021_paper.pdf)] [[Code](https://github.com/3DCVdeveloper/BGNet)]

* **RAFT-Stereo**: RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching, *3DV, 2021*. [[Paper](https://arxiv.org/abs/2109.07547)] [[Code](https://github.com/princeton-vl/RAFT-Stereo)]

* **CoEX**: Correlate-and-Excite: Real-Time Stereo Matching via Guided Cost Volume Excitation, *IROS, 2021*. [[Paper](https://antabangun.github.io/projects/CoEx/)] [[Code](https://github.com/antabangun/coex)]

* **SCV-Stereo**: SCV-Stereo: Learning Stereo Matching from a Sparse Cost Volume, *ICIP, 2021*. [[Paper](https://arxiv.org/abs/2107.08187)] [[Code](https://sites.google.com/view/scv-stereo)]

* **Separable-Stereo**: Separable Convolutions for Optimizing 3D Stereo Networks, *ICIP, 2021*. [[Paper](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/kognitive-systeme/projects/deepstereovision/)] [[Code](https://github.com/cogsys-tuebingen/separable-3D-convs-for-stereo-matching)]

* **CREStereo**: Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Practical_Stereo_Matching_via_Cascaded_Recurrent_Network_With_Adaptive_Correlation_CVPR_2022_paper.pdf)] [[Code](https://github.com/megvii-research/CREStereo)]

* **PCWNet**: PCW-Net: Pyramid Combination and Warping Cost Volume for Stereo Matching, *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920280.pdf)] [[Code](https://github.com/gallenszl/PCWNet)]

* **EAI-Stereo**: EAI-Stereo: Error Aware Iterative Network for Stereo Matching, *ACCV, 2022*. [[Paper](https://openaccess.thecvf.com/content/ACCV2022/html/Zhao_EAI-Stereo_Error_Aware_Iterative_Network_for_Stereo_Matching_ACCV_2022_paper.html)] [[Code](https://github.com/smartadpole/EAI-Stereo)]

* **ACVNet**: Attention Concatenation Volume for Accurate and Efficient Stereo Matching, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Attention_Concatenation_Volume_for_Accurate_and_Efficient_Stereo_Matching_CVPR_2022_paper.pdf)] [[Code](https://github.com/gangweiX/ACVNet)]

* **MobileStereoNet**: MobileStereoNet: Towards Lightweight Deep Networks for Stereo Matching, *WACV, 2022*. [[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Shamsafar_MobileStereoNet_Towards_Lightweight_Deep_Networks_for_Stereo_Matching_WACV_2022_paper.pdf)] [[Code](https://github.com/cogsys-tuebingen/mobilestereonet)]

* **GMStereo**: Unifying Flow, Stereo and Depth Estimation, *TPAMI, 2023*. [[Paper](https://arxiv.org/pdf/2211.05783.pdf)] [[Code](https://haofeixu.github.io/unimatch/)]

* **IGEV-Stereo**: Iterative Geometry Encoding Volume for Stereo Matching, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Iterative_Geometry_Encoding_Volume_for_Stereo_Matching_CVPR_2023_paper.pdf)] [[Code](https://github.com/gangweiX/IGEV)]

* **DLNR**: High-Frequency Stereo Matching Network, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_High-Frequency_Stereo_Matching_Network_CVPR_2023_paper.pdf)] [[Code](https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network)]

* **CroCo v2**: CroCo v2: Improved Cross-View Completion Pre-training for Stereo Matching and Optical Flow, *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Weinzaepfel_CroCo_v2_Improved_Cross-view_Completion_Pre-training_for_Stereo_Matching_and_ICCV_2023_paper.pdf)] [[Code](https://github.com/naver/croco)]

* **PCVNet**: Parameterized Cost Volume for Stereo Matching, *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zeng_Parameterized_Cost_Volume_for_Stereo_Matching_ICCV_2023_paper.pdf)] [[Code](https://github.com/jiaxiZeng/Parameterized-Cost-Volume-for-Stereo-Matching)]

* **CREStereo++**: Uncertainty Guided Adaptive Warping for Robust and Efficient Stereo Matching, *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Jing_Uncertainty_Guided_Adaptive_Warping_for_Robust_and_Efficient_Stereo_Matching_ICCV_2023_paper.pdf)]

* **ICVP**: Image-Coupled Volume Propagation for Stereo Matching, *ICIP, 2023*. [[Paper](https://ieeexplore.ieee.org/abstract/document/10222247)] [[Code](https://github.com/ohkwon718/icvp)]

* **TemporalStereo**: TemporalStereo: Efficient Spatial-Temporal Stereo Matching Network, *IROS, 2023*. [[Paper](https://youmi-zym.github.io/projects/TemporalStereo/)] [[Code](https://github.com/youmi-zym/TemporalStereo)]

* **UCFNet**: Digging Into Uncertainty-Based Pseudo-Label for Robust Stereo Matching, *TPAMI, 2023*. [[Paper](https://arxiv.org/pdf/2307.16509.pdf)] [[Code](https://github.com/gallenszl/UCFNet?tab=readme-ov-file)]

* **Selective-IGEV**: Selective-Stereo: Adaptive Frequency Information Selection for Stereo Matching, *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2403.00486.pdf)] [[Code](https://github.com/Windsrain/Selective-Stereo)]

* **GOAT**: Global Occlusion-Aware Transformer for Robust Stereo Matching, *WACV, 2024*. [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Global_Occlusion-Aware_Transformer_for_Robust_Stereo_Matching_WACV_2024_paper.pdf)] [[Code](https://github.com/Magicboomliu/GOAT)]


## Event Stereo

* **Event-IntensityStereo**: Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds, *ICCV, 2021*. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Mostafavi_Event-Intensity_Stereo_Estimating_Depth_by_the_Best_of_Both_Worlds_ICCV_2021_paper.pdf)] [[Code](https://github.com/yonseivnl/se-cff)]

* **SE-CFF**: Stereo Depth From Events Cameras: Concentrate and Focus on the Future, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Nam_Stereo_Depth_From_Events_Cameras_Concentrate_and_Focus_on_the_CVPR_2022_paper.pdf)] [[Code](https://github.com/yonseivnl/se-cff)]

* **SCSNet**: Selection and Cross Similarity for Event-Image Deep Stereo, *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920467.pdf)] [[Code](https://github.com/Chohoonhee/SCSNet)]

* **DTC-SPADE**: Discrete Time Convolution for Fast Event-Based Stereo, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Discrete_Time_Convolution_for_Fast_Event-Based_Stereo_CVPR_2022_paper.pdf)]

* **ADES**: Learning Adaptive Dense Event Stereo From the Image Domain, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Cho_Learning_Adaptive_Dense_Event_Stereo_From_the_Image_Domain_CVPR_2023_paper.pdf)]

* **SAFE**: Depth From Asymmetric Frame-Event Stereo: A Divide-and-Conquer Approach, *WACV, 2024*. [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Chen_Depth_From_Asymmetric_Frame-Event_Stereo_A_Divide-and-Conquer_Approach_WACV_2024_paper.pdf)]


## Talks
