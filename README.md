# awesome-Stereo-Matching
Welcome to the "awesome-Stereo" repository, a curated list of state-of-the-art stereo matching resources maintained by Fabio Tosi and Matteo Poggi. This repository aims to provide a comprehensive collection of the latest and most influential papers on stereo matching published in top-tier computer vision conferences and prestigious journals.

## Table of Contents

1. [Survey](#survey)
2. [CodeBase](#codebase)
3. [Datasets](#datasets)
   <!--*- [Real-World (Passive)](#real-world-passive)
   - [Real-World (Multimodal)](#real-world-multimodal)
   - [Synthetic](#synthetic) --!>
4. [Papers](#papers)
   * [End-to-End Learning](#end-to-end-learning)
       <!--* [Stereo Networks](#stereo-networks)
      * [Addressing Stereo Over-Smoothing Issue](#addressing-stereo-over-smoothing-issue)
      * [Zero-shot Generalization](#zero-shot-generalization)
      * [Self-Supervised](#self-supervised)
      * [Online Continual Adaptation](#online-continual-adaptation)
      * [Offline Adaptation](#offline-adaptation) --!>
   - [Stereo Pipeline](#stereo-pipeline)
      <!--** [Matching Cost](#matching-cost)
      * [Optimization](#optimization)
      * [Refinement](#refinement) --!>
   - [Event Stereo](#event-stereo)
5. [Talks & Tutorials](#talks)




## Survey

* *"Quantitative evaluation of confidence measures in a machine learning world"*, *ICCV, 2017*. [[Paper](https://arxiv.org/abs/2101.00431)] [[Bibtex](./bibliography/QuantitativeConf.txt)]

* *"On the synergies between machine learning and binocular stereo for depth estimation from images: a survey"*, *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2021*. [[Paper](https://arxiv.org/pdf/2004.08566.pdf)] [[Bibtex](./bibliography/OnTheSynergies.txt)]

* *"On the Confidence of Stereo Matching in a Deep-Learning Era: A Quantitative Evaluation"*, *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2022*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Poggi_Quantitative_Evaluation_of_ICCV_2017_paper.pdf)] [[Bibtex](./bibliography/OnTheConfidence.txt)]


## CodeBase

* **OpenStereo**: *"OpenStereo: A Comprehensive Benchmark for Stereo Matching and Strong Baseline"*, *arXiv, 2023* [[Code](https://github.com/XiandaGuo/OpenStereo)] [[Paper](https://arxiv.org/pdf/2312.00343.pdf)] [[Bibtex](./bibliography/OpenStereo.txt)]



## Datasets


<details><summary style="font-size: larger; font-weight: bold;">  Real-World</summary><ul>

   <details><summary> Passive </summary>
   
   * **Middlebury v3**: *"High-resolution stereo datasets with subpixel-accurate ground truth"*, *GCPR 2014*. [[Paper](https://elib.dlr.de/90624/1/ScharsteinEtal2014.pdf)] [[Dataset](https://vision.middlebury.edu/stereo/eval3/)] [[Bibtex](./bibliography/Middlebury_v3.txt)]

   * **KITTI 2012**: *"Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite"*, *CVPR, 2012*. [[Paper](https://projet.liris.cnrs.fr/imagine/pub/proceedings/CVPR2012/data/papers/424_O3C-04.pdf)] [[Dataset](https://www.cvlibs.net/datasets/kitti/)] [[Bibtex](./bibliography/KITTI_2012.txt)]

   * **KITTI 2015**: *"Object Scene Flow for Autonomous Vehicles"*, *CVPR, 2015*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Menze_Object_Scene_Flow_2015_CVPR_paper.pdf)] [[Dataset](https://www.cvlibs.net/datasets/kitti/)] [[Bibtex](./bibliography/KITTI_2015.txt)]

   * **ETH3D**: *"A multi-view stereo benchmark with high-resolution images and multi-camera videos"*, *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Schops_A_Multi-View_Stereo_CVPR_2017_paper.pdf)] [[Dataset](https://www.eth3d.net/)] [[Bibtex](./bibliography/ETH3D.txt)]


   * **Middlebury 2021 Mobile Dataset**: [[Website](https://vision.middlebury.edu/stereo/data/scenes2021/)] [[Bibtex](./bibliography/Middlebury_v3.txt)]

   * **The Booster Dataset**: *"Open Challenges in Deep Stereo: The Booster Dataset"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ramirez_Open_Challenges_in_Deep_Stereo_The_Booster_Dataset_CVPR_2022_paper.pdf)] [[Dataset](https://cvlab-unibo.github.io/booster-web/)] [[Bibtex](./bibliography/Booster.txt)]

   * **Holopix50k**: *"Holopix50k: A Large-Scale In-the-Wild Stereo Image Dataset"*, *CVPR, 2020*. [[Dataset](https://leiainc.github.io/holopix50k/)] [[Paper](https://arxiv.org/abs/2003.11172)] [[Bibtex](./bibliography/Holopix50k.txt)]

   * **InStereo2K**: *"InStereo2K: A Large Real Dataset for Stereo Matching in Indoor Scenes"*, *Science China Information Sciences, 2020*. [[Paper](https://link.springer.com/article/10.1007/s11432-019-2803-x)] [[Github](https://github.com/YuhuaXu/StereoDataset)]
   </details>

<details><summary> Multimodal </summary>

   * **DSEC**: *"DSEC: A Stereo Event Camera Dataset for Driving Scenarios"*, *RAL, 2021*. [[Paper](https://rpg.ifi.uzh.ch/docs/RAL21_DSEC.pdf)] [[Code](https://github.com/uzh-rpg/DSEC)] [[Dataset](https://dsec.ifi.uzh.ch/)] [[Bibtex](./bibliography/DSEC.txt)]

   * **Gated Stereo**: *"Gated Stereo: Joint Depth Estimation from Gated and Wide-Baseline Active Stereo Cues"*, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Walz_Gated_Stereo_Joint_Depth_Estimation_From_Gated_and_Wide-Baseline_Active_CVPR_2023_paper.pdf)] [[Dataset](https://light.princeton.edu/gatedstereo/)] [[Bibtex](./bibliography/Gated.txt)]

   * **RGB-MS**: *"RGB-Multispectral Matching: Dataset, Learning Methodology, Evaluation"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tosi_RGB-Multispectral_Matching_Dataset_Learning_Methodology_Evaluation_CVPR_2022_paper.pdf)] [[Dataset](https://cvlab-unibo.github.io/rgb-ms-web/)] [[Bibtex](./bibliography/RGB-MS.txt)]

   * **MS^2**: *"Deep Depth Estimation From Thermal Image"*, *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Shin_Deep_Depth_Estimation_From_Thermal_Image_CVPR_2023_paper.pdf)] [[Dataset](https://sites.google.com/view/multi-spectral-stereo-dataset)] [[Bibtex](./bibliography/MS2.txt)] 

</details>

<details><summary> Rendered </summary>

   * **The NeRF-Stereo Dataset**: *"NeRF-Supervised
   Deep Stereo"*, *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tosi_NeRF-Supervised_Deep_Stereo_CVPR_2023_paper.pdf)] [[Dataset](https://amsacta.unibo.it/id/eprint/7218/)] [[Bibtex](./bibliography/NS-Stereo.txt)] 

</details>

</ul>
</details>



<details open>
<summary style="font-size: larger; font-weight: bold;">Synthetic</summary>

* **Freiburg SceneFlow**: *"A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation"*, *CVPR, 2016*. [[Paper](https://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16/paper-MIFDB16.pdf)] [[Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)] [[Bibtex](./bibliography/SceneFlow.txt)]

* **HS-VS**: *"Hierarchical deep stereo matching on high-resolution image"*, *CVPR, 2019*. [[Paper](https://arxiv.org/pdf/1912.06704.pdf)] [[Dataset](https://github.com/gengshan-y/high-res-stereo?tab=readme-ov-file)] [[Bibtex](./bibliography/HSMNet.txt)]

* **TartanAir**: *"TartanAir: A dataset to push the limits of visual slam"*, *IROS, 2020*. [[Paper](https://ieeexplore.ieee.org/document/9341801?denied=)] [[Dataset](https://theairlab.org/tartanair-dataset/shorter)] [[Bibtex](./bibliography/TartanAir.txt)]

* **UnrealStereo4K**: *"SMD-Nets: Stereo Mixture Density Networks"*, *CVPR, 2021*. [[Paper](http://www.cvlibs.net/publications/Tosi2021CVPR.pdf)] [[Dataset](https://github.com/fabiotosi92/SMD-Nets)] [[Bibtex](./bibliography/SMD-Nets.txt)]


* **CREStereo**: *"Practical stereo matching via cascaded recurrent network with adaptive correlation"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Practical_Stereo_Matching_via_Cascaded_Recurrent_Network_With_Adaptive_Correlation_CVPR_2022_paper.pdf)] [[Dataset](https://github.com/megvii-research/CREStereo)] [[Bibtex](./bibliography/CreStereo.txt)]


* **SimStereo**: *"Active-Passive SimStereo – Benchmarking the Cross-Generalization Capabilities of Deep Learning-based Stereo Methods"*, *NeurIPS, 2022*. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/bc3a68a20e5c8ba5cbefc1ecf74bfaaa-Paper-Datasets_and_Benchmarks.pdf)] [[Dataset](https://ieee-dataport.org/open-access/active-passive-simstereo)] [[Bibtex](./bibliography/SimStereo.txt)]


* **Spring**: *"Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo"*, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Mehl_Spring_A_High-Resolution_High-Detail_Dataset_and_Benchmark_for_Scene_Flow_CVPR_2023_paper.html)] [[Dataset](http://spring-benchmark.org)] [[Bibtex](./bibliography/Spring.txt)]

* **Dynamic Replica**: *"DynamicStereo: Consistent Dynamic Depth From Stereo Videos"*, *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Karaev_DynamicStereo_Consistent_Dynamic_Depth_From_Stereo_Videos_CVPR_2023_paper.pdf)] [[Dataset](https://dynamic-stereo.github.io/)] [[Bibtex](./bibliography/Dynamic_Replica.txt)]

</details>

## Papers

### End-to-End Learning

<details open>
<summary style="font-size: larger; font-weight: bold;">Vanilla Stereo Networks</summary><ul>

  <details open>
    <summary style="font-size: larger; font-weight: bold;">2D Architectures</summary>
    
   * **DispNet-C**: *"A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation"*, *CVPR, 2016*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Mayer_A_Large_Dataset_CVPR_2016_paper.pdf)] [[Bibtex](./bibliography/SceneFlow.txt)]

   * **CNN+CRF**: *"End-to-end training of hybrid CNN-CRF models for stereo"*, *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Knobelreiter_End-To-End_Training_of_CVPR_2017_paper.pdf)]  [[Code](https://github.com/tuananh1007/End-to-End-Training-of-Hybrid-CNN-CRF-Models-for-Stereo)]  [[Bibtex](./bibliography/SceneFlow.txt)]

   * **CRL**: *"Cascade residual learning: A two-stage convolutional neural network for stereo matching"*, *CVPRW, 2017*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Pang_Cascade_Residual_Learning_ICCV_2017_paper.pdf)]  [[Code](https://github.com/jiahaopang/crl)]  [[Bibtex](./bibliography/CRL.txt)]

   * **iResNet**: *"Learning for disparity estimation through feature constancy"*, *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liang_Learning_for_Disparity_CVPR_2018_paper.pdf)]  [[Code](https://github.com/leonzfa/iResNet)]  [[Bibtex](./bibliography/iResNet.txt)]


   * **DispNet-CSS**: *"Occlusions, motion and depth boundaries with a generic network for disparity, optical flow or scene flow estimation"*, *ECCV, 2018*. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Eddy_Ilg_Occlusions_Motion_and_ECCV_2018_paper.pdf)]  [[Code](https://github.com/lmb-freiburg/netdef_models)]  [[Bibtex](./bibliography/DispNet-CSS.txt)]

   * **AutoDispNet-CSS**: *"Autodispnet: Improving disparity estimation with automl"*, *ICCV, 2019*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Saikia_AutoDispNet_Improving_Disparity_Estimation_With_AutoML_ICCV_2019_paper.pdf)]  [[Code](https://github.com/lmb-freiburg/autodispnet)]  [[Bibtex](./bibliography/AutoDispNet-CSS.txt)]

   * **HD<sup>3**: *"Hierarchical discrete distribution decomposition for match density estimation"*, *ICCV, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yin_Hierarchical_Discrete_Distribution_Decomposition_for_Match_Density_Estimation_CVPR_2019_paper.pdf)]  [[Code](https://github.com/ucbdrive/hd3)]  [[Bibtex](./bibliography/HD3.txt)]

   * **EdgeStereo**: *"Edgestereo: A context integrated residual pyramid network for stereo matching"*, *ACCV, 2018*. [[Paper](https://arxiv.org/pdf/1803.05196.pdf)]   [[Bibtex](./bibliography/Edgestereo.txt)]



  </details>


  <details open class="nested-details">
    <summary style="font-size: larger; font-weight: bold;">3D Architectures</summary>

   * **GC-Net**: *"End-to-end learning of geometry and context for deep stereo regression"*, *ICCV, 2017*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Kendall_End-To-End_Learning_of_ICCV_2017_paper.pdf)] [[Bibtex](./bibliography/GC-Net.txt)]

   * **HSMNet**: *"Hierarchical deep stereo matching on high-resolution images"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Hierarchical_Deep_Stereo_Matching_on_High-Resolution_Images_CVPR_2019_paper.pdf)] [[Code](https://github.com/gengshan-y/high-res-stereo)]  [[Bibtex](./bibliography/HSMNet.txt)]

   * **ECA**: *"Deep stereo matching with explicit cost aggregation sub-architecture"*, *AAAI, 2018*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/12267/12126)] [[Bibtex](./bibliography/ECA.txt)]

   * **PSMNet**: *"Pyramid Stereo Matching Network"*, *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chang_Pyramid_Stereo_Matching_CVPR_2018_paper.pdf)] [[Code](https://github.com/JiaRenChang/PSMNet)] [[Bibtex](./bibliography/PSMNet.txt)]

   * **EMCUA**: *"Pyramid Stereo Matching Network"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Nie_Multi-Level_Context_Ultra-Aggregation_for_Stereo_Matching_CVPR_2019_paper.pdf)] [[Bibtex](./bibliography/EMCUA.txt)]

  </details>


  <details open class="nested-details">
    <summary style="font-size: larger; font-weight: bold;">GRU-based Architectures</summary>
    
   * **RAFT-Stereo**: *"RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching"*, *3DV, 2021*. [[Paper](https://arxiv.org/abs/2109.07547)] [[Code](https://github.com/princeton-vl/RAFT-Stereo)] [[Bibtex](./bibliography/RAFT-Stereo.txt)]

   * **CREStereo**: *"Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Practical_Stereo_Matching_via_Cascaded_Recurrent_Network_With_Adaptive_Correlation_CVPR_2022_paper.pdf)] [[Code](https://github.com/megvii-research/CREStereo)] [[Bibtex](./bibliography/CREStereo.txt)]

  </details>

  <details open class="nested-details">
    <summary style="font-size: larger; font-weight: bold;">Transformer-based Architectures</summary>
    
   * **STTR**: *"Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective With Transformers"*, *ICCV, 2021*  [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Revisiting_Stereo_Depth_Estimation_From_a_Sequence-to-Sequence_Perspective_With_Transformers_ICCV_2021_paper.pdf)] [[Code](https://github.com/mli0603/stereo-transformer)] [[Bibtex](./bibliography/STTR.txt)]

   * **CEST**: *"Context-enhanced stereo transformer"*, *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920263.pdf)] [[Code](https://github.com/guoweiyu/Context-Enhanced-Stereo-Transformer)] [[Bibtex](./bibliography/CEST.txt)]

   * **Chitransformer**: *"Chitransformer: Towards Reliable Stereo From Cues"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Su_Chitransformer_Towards_Reliable_Stereo_From_Cues_CVPR_2022_paper.pdf)] [[Code](https://github.com/ISL-CV/ChiTransformer)] [[Bibtex](./bibliography/Chitransformer.txt)]

   * **GMStereo**: *"Unifying Flow, Stereo and Depth Estimation"*, *TPAMI, 2023*. [[Paper](https://arxiv.org/pdf/2211.05783.pdf)] [[Code](https://haofeixu.github.io/unimatch/)] [[Bibtex](./bibliography/GMStereo.txt)]

   * **CroCo v2**: *"CroCo v2: Improved Cross-View Completion Pre-training for Stereo Matching and Optical Flow"*, *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Weinzaepfel_CroCo_v2_Improved_Cross-view_Completion_Pre-training_for_Stereo_Matching_and_ICCV_2023_paper.pdf)] [[Code](https://github.com/naver/croco)] [[Bibtex](./bibliography/CroCo.txt)]

   * **GOAT**: *"Global Occlusion-Aware Transformer for Robust Stereo Matching"*, *WACV, 2024*. [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Global_Occlusion-Aware_Transformer_for_Robust_Stereo_Matching_WACV_2024_paper.pdf)] [[Code](https://github.com/Magicboomliu/GOAT)] [[Bibtex](./bibliography/GOAT.txt)]

  </details>



* **LEAStereo**: *"Hierarchical Neural Architecture Search for Deep Stereo Matching"*, *NeurIPS, 2020*. [[Paper](https://arxiv.org/pdf/2010.13501.pdf)] [[Code](https://github.com/XuelianCheng/LEAStereo)] [[Bibtex](./bibliography/LEAStereo.txt)]


* **AANet**: *"AANet: Adaptive Aggregation Network for Efficient Stereo Matching"*, *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_AANet_Adaptive_Aggregation_Network_for_Efficient_Stereo_Matching_CVPR_2020_paper.pdf)] [[Code](https://github.com/haofeixu/aanet)] [[Bibtex](./bibliography/AANet.txt)]

* **CasStereo**: *"Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching"*, *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gu_Cascade_Cost_Volume_for_High-Resolution_Multi-View_Stereo_and_Stereo_Matching_CVPR_2020_paper.pdf)] [[Code](https://github.com/alibaba/cascade-stereo)]  [[Bibtex](./bibliography/CasStereo.txt)]

* **Bi3D**: *"Bi3D: Stereo Depth Estimation via Binary Classifications"*, *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Badki_Bi3D_Stereo_Depth_Estimation_via_Binary_Classifications_CVPR_2020_paper.pdf)] [[Code](https://github.com/NVlabs/Bi3D)] [[Bibtex](./bibliography/Bi3D.txt)]

* **WaveletStereo**: *"WaveletStereo: Learning Wavelet Coefficients of Disparity Map in Stereo Matching"*, *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_WaveletStereo_Learning_Wavelet_Coefficients_of_Disparity_Map_in_Stereo_Matching_CVPR_2020)] [[Bibtex](./bibliography/WaveletStereo.txt)]

* **MABNet**: *"MABNet: a lightweight stereo network based on multibranch adjustable bottleneck module"*, *ECCV, 2020*. [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730341.pdf)] [[Code](https://github.com/JumpXing/MABNet)] [[Bibtex](./bibliography/MABNet.txt)] 


* **Fadnet**: *"Fadnet: A Fast and Accurate Network for Disparity Estimation"*, *ICRA, 2020*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9197031)] [[Code](https://github.com/HKBU-HPML/FADNet)] [[Bibtex](./bibliography/Fadnet.txt)]

* **SGNet**: *"SGNet: Semantics Guided Deep Stereo Matching"*, *ACCV, 2020*. [[Paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Chen_SGNet_Semantics_Guided_Deep_Stereo_Matching_ACCV_2020_paper.pdf)] [[Bibtex](./bibliography/Fadnet.txt)] [[Bibtex](./bibliography/SGNet.txt)]

* **AAFS**: *"Attention-Aware Feature Aggregation for Real-time Stereo Matching on Edge Devices"*, *ACCV, 2020* [[Code](https://github.com/JiaRenChang/RealtimeStereo)] [[Paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Chang_Attention-Aware_Feature_Aggregation_for_Real-time_Stereo_Matching_on_Edge_Devices_ACCV_2020_paper.pdf)] [[Bibtex](./bibliography/AAFS.txt)]

* **Fast DS-CS**: *"Fast Deep Stereo with 2D Convolutional Processing of Cost Signatures"*, *WACV, 2020* [[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Yee_Fast_Deep_Stereo_with_2D_Convolutional_Processing_of_Cost_Signatures_WACV_2020_paper.pdf)]  [[Code](https://github.com/ayanc/fdscs)]  [[Bibtex](./bibliography/FDCSC.txt)]

* **CFNet**: *"CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching"*, *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Shen_CFNet_Cascade_and_Fused_Cost_Volume_for_Robust_Stereo_Matching_CVPR_2021_paper.pdf)] [[Code](https://github.com/gallenszl/CFNet)] [[Bibtex](./bibliography/CFNet.txt)]

* **HITNet**: *"HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching"*, *CVPR, 2021*. [[Paper](https://arxiv.org/abs/2007.12140)] [[Code](https://github.com/google-research/google-research/tree/master/hitnet)] [[Bibtex](./bibliography/HITNet.txt)]

* **BGNet**: *"Bilateral Grid Learning for Stereo Matching Networks"*, *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Bilateral_Grid_Learning_for_Stereo_Matching_Networks_CVPR_2021_paper.pdf)] [[Code](https://github.com/3DCVdeveloper/BGNet)] [[Bibtex](./bibliography/BGNet.txt)]



* **UASNet**: *"UASNet: Uncertainty Adaptive Sampling Network for Deep Stereo Matching"*, *ICCV, 2021* [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9710298)] [[Bibtex](./bibliography/UASNet.txt)]



* **CoEX**: *"Correlate-and-Excite: Real-Time Stereo Matching via Guided Cost Volume Excitation"*, *IROS, 2021*. [[Paper](https://antabangun.github.io/projects/CoEx/)] [[Code](https://github.com/antabangun/coex)] [[Bibtex](./bibliography/CoEX.txt)]

* **SCV-Stereo**: *"SCV-Stereo: Learning Stereo Matching from a Sparse Cost Volume"*, *ICIP, 2021*. [[Paper](https://arxiv.org/abs/2107.08187)] [[Code](https://sites.google.com/view/scv-stereo)] [[Bibtex](./bibliography/SCV-Stereo.txt)] 

* **Separable-Stereo**: *"Separable Convolutions for Optimizing 3D Stereo Networks"*, *ICIP, 2021*. [[Paper](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/kognitive-systeme/projects/deepstereovision/)] [[Code](https://github.com/cogsys-tuebingen/separable-3D-convs-for-stereo-matching)] [[Bibtex](./bibliography/Separable-Stereo.txt)]


* **PCWNet**: *"PCW-Net: Pyramid Combination and Warping Cost Volume for Stereo Matching"*, *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920280.pdf)] [[Code](https://github.com/gallenszl/PCWNet)] [[Bibtex](./bibliography/PCWNet.txt)]



* **EASNet**: *"EASNet: searching elastic and accurate network architecture for stereo matching"*, *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920434.pdf)] [[Code](https://github.com/HKBU-HPML/EASNet)] [[Bibtex](./bibliography/EASNet.txt)]

* **EAI-Stereo**: *"EAI-Stereo: Error Aware Iterative Network for Stereo Matching"*, *ACCV, 2022*. [[Paper](https://openaccess.thecvf.com/content/ACCV2022/html/Zhao_EAI-Stereo_Error_Aware_Iterative_Network_for_Stereo_Matching_ACCV_2022_paper.html)] [[Code](https://github.com/smartadpole/EAI-Stereo)] [[Bibtex](./bibliography/EAI-Stereo.txt)]

* **PBCStereo**: *"PBCStereo: A Compressed Stereo Network with Pure Binary Convolutional Operations"*, *ACCV, 2022*. [[Paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Cai_PBCStereo_A_Compressed_Stereo_Network_with_Pure_Binary_Convolutional_Operations_ACCV_2022_paper.pdf)] [[Bibtex](./bibliography/PBCStereo.txt)]

* **ACVNet**: *"Attention Concatenation Volume for Accurate and Efficient Stereo Matching"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Attention_Concatenation_Volume_for_Accurate_and_Efficient_Stereo_Matching_CVPR_2022_paper.pdf)] [[Code](https://github.com/gangweiX/ACVNet)] [[Bibtex](./bibliography/ACVNet.txt)]

* **MobileStereoNet**: *"MobileStereoNet: Towards Lightweight Deep Networks for Stereo Matching"*, *WACV, 2022*. [[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Shamsafar_MobileStereoNet_Towards_Lightweight_Deep_Networks_for_Stereo_Matching_WACV_2022_paper.pdf)] [[Code](https://github.com/cogsys-tuebingen/mobilestereonet)] [[Bibtex](./bibliography/MobileStereoNet.txt)]


* **IGEV-Stereo**: *"Iterative Geometry Encoding Volume for Stereo Matching"*, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Iterative_Geometry_Encoding_Volume_for_Stereo_Matching_CVPR_2023_paper.pdf)] [[Code](https://github.com/gangweiX/IGEV)] [[Bibtex](./bibliography/IGEV-Stereo.txt)]

* **DLNR**: *"High-Frequency Stereo Matching Network"*, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_High-Frequency_Stereo_Matching_Network_CVPR_2023_paper.pdf)] [[Code](https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network)] [[Bibtex](./bibliography/DLNR.txt)]



* **PCVNet**: *"Parameterized Cost Volume for Stereo Matching"*, *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zeng_Parameterized_Cost_Volume_for_Stereo_Matching_ICCV_2023_paper.pdf)] [[Code](https://github.com/jiaxiZeng/Parameterized-Cost-Volume-for-Stereo-Matching)] [[Bibtex](./bibliography/PCVNet.txt)]

* **CREStereo++**: *"Uncertainty Guided Adaptive Warping for Robust and Efficient Stereo Matching"*, *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Jing_Uncertainty_Guided_Adaptive_Warping_for_Robust_and_Efficient_Stereo_Matching_ICCV_2023_paper.pdf)] [[Bibtex](./bibliography/CREStereo++.txt)]

* **ELFNet**: *"Elfnet: Evidential local-global fusion for stereo matching"*, *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Su_Chitransformer_Towards_Reliable_Stereo_From_Cues_CVPR_2022_paper.pdf)] [[Code](https://openaccess.thecvf.com/content/ICCV2023/papers/Lou_ELFNet_Evidential_Local-global_Fusion_for_Stereo_Matching_ICCV_2023_paper.pdf)] [[Bibtex](./bibliography/ELFNet.txt)]

* **ICVP**: *"Image-Coupled Volume Propagation for Stereo Matching"*, *ICIP, 2023*. [[Paper](https://ieeexplore.ieee.org/abstract/document/10222247)] [[Code](https://github.com/ohkwon718/icvp)] [[Bibtex](./bibliography/ICVP.txt)]

* **TemporalStereo**: *"TemporalStereo: Efficient Spatial-Temporal Stereo Matching Network"*, *IROS, 2023*. [[Paper](https://youmi-zym.github.io/projects/TemporalStereo/)] [[Code](https://github.com/youmi-zym/TemporalStereo)]  [[Bibtex](./bibliography/TemporalStereo.txt)]

* **StereoVAE**: *"StereoVAE: A lightweight stereo-matching system using embedded GPUs"*, *ICRA, 2023*. [[Paper](https://ieeexplore.ieee.org/document/10160441)] [[Bibtex](./bibliography/StereoVAE.txt)]

* **UCFNet**: *"Digging Into Uncertainty-Based Pseudo-Label for Robust Stereo Matching"*, *TPAMI, 2023*. [[Paper](https://arxiv.org/pdf/2307.16509.pdf)] [[Code](https://github.com/gallenszl/UCFNet?tab=readme-ov-file)] [[Bibtex](./bibliography/UCFNet.txt)]

* **Selective-IGEV**: *"Selective-Stereo: Adaptive Frequency Information Selection for Stereo Matching"*, *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2403.00486.pdf)] [[Code](https://github.com/Windsrain/Selective-Stereo)] [[Bibtex](./bibliography/Selective-IGEV.txt)]



* **NMRF**: *"Neural Markov Random Field for Stereo Matching"*, *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2403.11193.pdf)] [[Code](https://github.com/aeolusguan/NMRF)] [[Bibtex](./bibliography/NMRF.txt)]

</ul>
</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Addressing Stereo Over-Smoothing Issue</summary>

* **SM-CDE**: *"On the over-smoothing problem of cnn based disparity estimation"*, *ICCV, 2019*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_On_the_Over-Smoothing_Problem_of_CNN_Based_Disparity_Estimation_ICCV_2019_paper.html)] [[Bibtex](./bibliography/SM-CDE.txt)] 

* **AcfNet**: *"Adaptive Unimodal Cost Volume Filtering for Deep Stereo Matching"*, *AAAI, 2020*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6991/6845)] [[Code](https://github.com/youmi-zym/AcfNet)] [[Bibtex](./bibliography/WaveletStereo.txt)]


* **CDN**: *"Wasserstein Distances for Stereo Disparity Estimation"*, *NeurIPS, 2020*. [[Paper](https://papers.nips.cc/paper/2020/file/fe7ecc4de28b2c83c016b5c6c2acd826-Paper.pdf)] [[Code](https://github.com/Div99/W-Stereo-Disp)] [[Bibtex](./bibliography/CDN.txt)] 

* **SMD-Nets**: *"SMD-Nets: Stereo Mixture Density Networks"*, *CVPR, 2021*. [[Paper](http://www.cvlibs.net/publications/Tosi2021CVPR.pdf)] [[Code](https://github.com/fabiotosi92/SMD-Nets)] [[Bibtex](./bibliography/SMD-Nets.txt)] 

* **NDR**: "*Neural disparity refinement for arbitrary resolution stereo*", *3DV, 2021*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9665913?casa_token=3rm4WpqLb_QAAAAA:5Sa0RO547j8LsaEYUeppzB33gZJg5Y3tfiPVwM9rzs9MEAuoHSta0Kdw3Cm9NrtfOOdFkIwp)] [[Website](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR.txt)] 

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Zero-shot Generalization</summary>


* **DSM-Net**: "*Domain-invariant Stereo Matching Networks*", *ECCV, 2020*. [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470409.pdf)] [[Code](https://github.com/feihuzhang/DSMNet)] [[Bibtex](./bibliography/DSM-Net.txt)] 

* **StereoGAN**: "*StereoGAN: Bridging Synthetic-to-Real Domain Gap by Joint Optimization of Domain Translation and Stereo Matching*", *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_StereoGAN_Bridging_Synthetic-to-Real_Domain_Gap_by_Joint_Optimization_of_Domain_CVPR_2020_paper.pdf)] [[Code](https://github.com/ruiliu-ai/StereoGAN)] [[Bibtex](./bibliography/StereoGAN.txt)] 

* **MS-Nets**: "*Matching-space Stereo Networks for Cross-domain Generalization*", *3DV, 2020*. [[Paper](https://mordohai.github.io/public/Cai_MatchingSpaceStereo20.pdf)] [[Code](https://github.com/ccj5351/MS-Nets)] [[Bibtex](./bibliography/MS-Nets.txt)] 

* **NDR**: "*Neural disparity refinement for arbitrary resolution stereo*", *3DV, 2021*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9665913?casa_token=3rm4WpqLb_QAAAAA:5Sa0RO547j8LsaEYUeppzB33gZJg5Y3tfiPVwM9rzs9MEAuoHSta0Kdw3Cm9NrtfOOdFkIwp)] [[Website](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR.txt)] 

* **ARStereo**: "*Revisiting Non-Parametric Matching Cost Volumes for Robust and Generalizable Stereo Matching*", *NeurIPS, 2022*. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/6794f555524c9069e26970a408d353cc-Paper-Conference.pdf)] [[Code](https://github.com/kelkelcheng/AdversariallyRobustStereo)]  [[Bibtex](./bibliography/ARStereo.txt)] 

* **FCStereo**: "*Revisiting Domain Generalized Stereo Matching Networks From a Feature Consistency Perspective*", *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Revisiting_Domain_Generalized_Stereo_Matching_Networks_From_a_Feature_Consistency_CVPR_2022_paper.pdf)] [[Code](https://github.com/jiaw-z/FCStereo)] [[Bibtex](./bibliography/FCStereo.txt)] 

* **GraftNet**: "*GraftNet: Towards Domain Generalized Stereo Matching With a Broad-Spectrum and Task-Oriented Feature*", *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_GraftNet_Towards_Domain_Generalized_Stereo_Matching_With_a_Broad-Spectrum_and_CVPR_2022_paper.pdf)] [[Code](https://github.com/SpadeLiu/Graft-PSMNet)] [[Bibtex](./bibliography/GraftNet.txt)] 

* **ITSA**: "*ITSA: An Information-Theoretic Approach to Automatic Shortcut Avoidance and Domain Generalization in Stereo Matching Networks*", *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chuah_ITSA_An_Information-Theoretic_Approach_to_Automatic_Shortcut_Avoidance_and_Domain_CVPR_2022_paper.pdf)] [[Code](https://github.com/waychin-weiqin/ITSA)] [[Bibtex](./bibliography/ITSA.txt)] 

* **EVHS**: "*Expansion of Visual Hints for Improved Generalization in Stereo Matching*", *WACV, 2023*. [[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Pilzer_Expansion_of_Visual_Hints_for_Improved_Generalization_in_Stereo_Matching_WACV_2023_paper.pdf)] [[Bibtex](./bibliography/EVHS.txt)] 
 
* **LSSI**: "*Learning Stereo from Single Images*", *ECCV, 2020*. [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460698.pdf)] [[Code](https://github.com/nianticlabs/stereo-from-mono/)] [[Bibtex](./bibliography/LSSI.txt)] 

* **NeRF-Supervised Stereo**: "*NeRF-Supervised Deep Stereo*", *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tosi_NeRF-Supervised_Deep_Stereo_CVPR_2023_paper.pdf)] [[Website](https://nerfstereo.github.io/)] [[Code](https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo)] [[Bibtex](./bibliography/NS-Stereo.txt)] 

* **DKT-Stereo**: "*Robust Synthetic-to-Real Transfer for Stereo Matching*", *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2403.07705)] [[Code](https://github.com/jiaw-z/DKT-Stereo)] [[Bibtex](./bibliography/DKT-Stereo.txt)] 

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Self-Supervised</summary>

* **Flow2Stereo**: *"Flow2Stereo: Effective Self-Supervised Learning of Optical Flow and Stereo Matching"*, *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Flow2Stereo_Effective_Self-Supervised_Learning_of_Optical_Flow_and_Stereo_Matching_CVPR_2020_paper.pdf)] [[Code](https://github.com/ppliuboy/Flow2Stereo)] [[Bibtex](./bibliography/Flow2Stereo.txt)]

* **Reversing-Stereo**: *"Reversing the cycle: self-supervised deep stereo through enhanced monocular distillation"*, *ECCV, 2020*. [[Paper](https://arxiv.org/pdf/2008.07130.pdf)] [[Code](https://github.com/FilippoAleotti/Reversing)] [[Bibtex](./bibliography/Reversing-Stereo.txt)]

</details>


<details open>
<summary style="font-size: larger; font-weight: bold;">Offline Adaptation</summary>

* **AdaStereo**: *"AdaStereo: A Simple and Efficient Approach for Adaptive Stereo Matching"*, *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Song_AdaStereo_A_Simple_and_Efficient_Approach_for_Adaptive_Stereo_Matching_CVPR_2021_paper.pdf)] [[Bibtex](./bibliography/AdaStereo.txt)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Online Continual Adaptation</summary>

* **Continual Adaptation for Deep Stereo**: *"Continual adaptation for deep stereo"*, *TPAMI, 2021*. [[Paper](https://ieeexplore.ieee.org/document/9418523?denied=)] [[Code](https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo?tab=readme-ov-file)]

* **RAG**: *"Continual Stereo Matching of Continuous Driving Scenes With Growing Architecture"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Continual_Stereo_Matching_of_Continuous_Driving_Scenes_With_Growing_Architecture_CVPR_2022_paper.pdf)] [[Bibtex](./bibliography/RAG.txt)]

* **PointFix**: *"PointFix: Learning to Fix Domain Bias for Robust Online Stereo Adaptation"*, *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136980557.pdf)] [[Bibtex](./bibliography/PointFix.txt)]

* **FedStereo**: *"Federated Online Adaptation for Deep Stereo"*, *CVPR, 2024*. [[Paper](https://fedstereo.github.io/)] [[Bibtex](./bibliography/FedStereo.txt)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Semantic Stereo Matching</summary>

* **RSS-Net**: *"Real-time semantic stereo matching"*, *ICRA, 2020*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9196784/)] [[Bibtex](./bibliography/RSS-Net.txt)]


</details>

## Stereo Pipeline

<details open>
<summary style="font-size: larger; font-weight: bold;">Matching Cost</summary>


* **Deep Embed**: *"A deep visual correspondence embedding model for stereo matching costs"*, *ICCV, 2015*. [[Paper](https://openaccess.thecvf.com/content_iccv_2015/papers/Chen_A_Deep_Visual_ICCV_2015_paper.pdf)] [[Bibtex](./bibliography/Deep_Embed.txt)]

* **MC-CNN**: *"Stereo matching by training a convolutional neural network to compare image patches"*, *JMLR, 2016*. [[Paper](https://www.jmlr.org/papers/v17/15-535.html)] [[Code](https://github.com/jzbontar/mc-cnn)] [[Bibtex](./bibliography/MC-CNN.txt)]

* **Content CNN**: *"Efficient deep learning for stereo matching"*, *CVPR, 2016*. [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Luo_Efficient_Deep_Learning_CVPR_2016_paper.pdf)] [[Code](https://github.com/datvuthanh/Stereo-Matching)] [[Bibtex](./bibliography/Content_CNN.txt)]

* **Per-pixel pyramid-pooling**: *"Look wider to match image patches with convolutional neural networks"*, *SPR, 2016*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7778222)] [[Bibtex](./bibliography/per-pixel_pyramid-pooling.txt)]

* **Consistency and Distinctiveness**: *"Fundamental principles on learning new features for effective dense matching"*, *TIP, 2017*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8038003)] [[Bibtex](./bibliography/Consistency_and_Distinctiveness.txt)]

* **MC-CNN-WS**: *"Weakly supervised learning of deep metrics for stereo reconstruction"*, *ICCV, 2017*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tulyakov_Weakly_Supervised_Learning_ICCV_2017_paper.pdf)] [[Code](https://github.com/tlkvstepan/mc-cnn-ws)] [[Bibtex](./bibliography/MC-CNN-WS.txt)]

* **CBMV**: *"CBMV: A coalesced bidirectional matching volume for disparity estimation"*, *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Batsos_CBMV_A_Coalesced_CVPR_2018_paper.pdf)] [[Bibtex](./bibliography/CBMV.txt)]

* **SDC**: *"SDC - stacked dilated convolution: A unified descriptor network for dense matching tasks"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Schuster_SDC_-_Stacked_Dilated_Convolution_A_Unified_Descriptor_Network_for_CVPR_2019_paper.pdf)] [[Bibtex](./bibliography/SDC.txt)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Optimization</summary>

* **GCP**: *"Learning to detect ground control points for improving the accuracy of stereo matching"*, *CVPR, 2014*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2014/papers/Spyropoulos_Learning_to_Detect_2014_CVPR_paper.pdf)] [[Bibtex](./bibliography/GCP.txt)]

* **LevStereo**: *"Leveraging stereo matching with learning-based confidence measures"*, *CVPR, 2015*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Park_Leveraging_Stereo_Matching_2015_CVPR_paper.pdf)] [[Bibtex](./bibliography/LevStereo.txt)]

* **O1**: *"Learning a general-purpose confidence measure based on o (1) features and a smarter aggregation strategy for semi global matching"*, *3DV, 2016*. [[Paper](http://vision.disi.unibo.it/~mpoggi/papers/3dv2016_o1.pdf)] [[Bibtex](./bibliography/O1.txt)]

* **PBCP**: *"Patch Based Confidence Prediction for Dense Disparity Map"*, *BMVC, 2016*. [[Paper](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Seki2016BMVC.pdf)] [[Bibtex](./bibliography/PBCP.txt)]

* **Sgm-Nets**: *"Sgm-Nets: Semi-global matching with neural networks"*, *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Seki_SGM-Nets_Semi-Global_Matching_CVPR_2017_paper.pdf)] [[Bibtex](./bibliography/Sgm-Nets.txt)]

* **SGM-Forest**: *"Learning to fuse proposals from multiple scanline optimizations in semi-global matching"*, *ECCV, 2018*. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Johannes_Schoenberger_Learning_to_Fuse_ECCV_2018_paper.pdf)] [[Bibtex](./bibliography/SGM-Forest.txt)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Refinement</summary>

* **GDN**: *"Improved stereo matching with constant highway networks and reflective confidence learning"*, *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Shaked_Improved_Stereo_Matching_CVPR_2017_paper.pdf)] [[Code](https://github.com/amitshaked/resmatch)] [[Bibtex](./bibliography/GDN.txt)]

* **DRR**: *"Detect, replace, refine: Deep structured prediction for pixel wise labeling"*, *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Gidaris_Detect_Replace_Refine_CVPR_2017_paper.pdf)] [[Code](https://github.com/gidariss/DRR_struct_pred/)] [[Bibtex](./bibliography/DRR.txt)]

* **OSD**: *"Efficient stereo matching leveraging deep local and context information"*, *IEEE Access, 2017*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8047938)] [[Bibtex](./bibliography/OSD.txt)]

* **Recresnet**: *"Recresnet: A recurrent residual cnn architecture for disparity map enhancement"*, *3DV, 2018*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8490974)] [[Code](https://github.com/kbatsos/RecResNet)] [[Bibtex](./bibliography/Recresnet.txt)]

* **LRCR**: *"Left-right comparative recurrent model for stereo matching"*, *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jie_Left-Right_Comparative_Recurrent_CVPR_2018_paper.pdf)] [[Bibtex](./bibliography/LRCR.txt)]

* **VRN**: *"Learned collaborative stereo refinement"*, *IJCV, 2021*. [[Paper](https://link.springer.com/article/10.1007/s11263-021-01485-5)] [[Bibtex](./bibliography/VRN.txt)]

* **FD-Fusion**: *"Fast stereo disparity maps refinement by fusion of data-based and model-based estimations"*, *3DV, 2019*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8886031)] [[Code](https://github.com/ferreram/FD-Fusion)] [[Bibtex](./bibliography/FD-Fusion.txt)]

* **NDR**: "*Neural disparity refinement for arbitrary resolution stereo*", *3DV, 2021*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9665913?casa_token=3rm4WpqLb_QAAAAA:5Sa0RO547j8LsaEYUeppzB33gZJg5Y3tfiPVwM9rzs9MEAuoHSta0Kdw3Cm9NrtfOOdFkIwp)] [[Website](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR.txt)] 

</details>

## Event Stereo
<details open>
<summary style="font-size: larger; font-weight: bold;">Event Stereo Networks</summary>

* **Event-IntensityStereo**: *"Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds"*, *ICCV, 2021*. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Mostafavi_Event-Intensity_Stereo_Estimating_Depth_by_the_Best_of_Both_Worlds_ICCV_2021_paper.pdf)] [[Code](https://github.com/yonseivnl/se-cff)] [[Bibtex](./bibliography/Event-IntensityStereo.txt)]

* **SE-CFF**: *"Stereo Depth From Events Cameras: Concentrate and Focus on the Future"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Nam_Stereo_Depth_From_Events_Cameras_Concentrate_and_Focus_on_the_CVPR_2022_paper.pdf)] [[Code](https://github.com/yonseivnl/se-cff)] [[Bibtex](./bibliography/SE-CFF.txt)]

* **SCSNet**: *"Selection and Cross Similarity for Event-Image Deep Stereo"*, *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920467.pdf)] [[Code](https://github.com/Chohoonhee/SCSNet)] [[Bibtex](./bibliography/SCSNet.txt)]

* **DTC-SPADE**: *"Discrete Time Convolution for Fast Event-Based Stereo"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Discrete_Time_Convolution_for_Fast_Event-Based_Stereo_CVPR_2022_paper.pdf)] [[Bibtex](./bibliography/DTC-SPADE.txt)]

* **ADES**: *"Learning Adaptive Dense Event Stereo From the Image Domain"*, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Cho_Learning_Adaptive_Dense_Event_Stereo_From_the_Image_Domain_CVPR_2023_paper.pdf)] [[Bibtex](./bibliography/ADES.txt)]

* **SAFE**: *"Depth From Asymmetric Frame-Event Stereo: A Divide-and-Conquer Approach"*, *WACV, 2024*. [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Chen_Depth_From_Asymmetric_Frame-Event_Stereo_A_Divide-and-Conquer_Approach_WACV_2024_paper.pdf)] [[Bibtex](./bibliography/SAFE.txt)]

</details>

## Talks & Tutorials

* **Facing depth estimation in-the-wild with deep networks**. M. Poggi, F. Tosi, F. Aleotti, K. Batsos, P. Mordohai, S. Mattoccia; ECCV 2020, SEC, Glasgow [[Website](https://sites.google.com/view/eccv-2020-robust-depth/home)]

* **Learning and understanding single image depth estimation in the wild**.  M. Poggi, F. Tosi, F. Aleotti, S. Mattoccia, C. Godard, J. Watson, M. Firman, G.J. Brostow; CVPR 2020, Seattle, Washington, US  [[Website](https://sites.google.com/view/cvpr-2020-depth-from-mono/home)]

* **Learning-based depth estimation from stereo and monocular images: successes, limitations and future challenges**. M. Poggi, F. Tosi, K. Batsos, P. Mordohai, S. Mattoccia, CVPR 2019, Long Beach, California, US [[Website](https://sites.google.com/view/cvpr-2019-depth-from-image/home)]

* **Learning-based depth estimation from stereo and monocular images: successes, limitations and future challenges**.  M. Poggi, F. Tosi, K. Batsos, P. Mordohai, S. Mattoccia; 3DV 2018, Verona, Italy [[Website](https://sites.google.com/view/3dv-2018-depth-from-image/home)]

* **Lecture: Computer Vision (Prof. Andreas Geiger, University of Tübingen)**. [[Preliminaries](https://www.youtube.com/watch?v=6hr6xpOU-uw&list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_&index=12&pp=iAQB)] [[Block Matching](https://www.youtube.com/watch?v=EVzEJQl8WFk&list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_&index=13&t=830s&pp=iAQB)] [[Siamese Networks](https://www.youtube.com/watch?v=vLgsiIXNf0I&list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_&index=14&pp=iAQB)] [[Spatial Regularization](https://www.youtube.com/watch?v=gqz6R1qChVQ&list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_&index=15&t=359s&pp=iAQB)] [[End-to-End Learning](https://www.youtube.com/watch?v=9vrmwZ9Pl4o&list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_&index=16&t=639s&pp=iAQB)] 


