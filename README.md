# Awesome-Deep-Stereo-Matching
Welcome to the "Awesome-Deep-Stereo-Matching" repository, a curated list of state-of-the-art deep stereo matching resources maintained by Fabio Tosi and Matteo Poggi. This repository aims to provide a comprehensive collection of the latest and most influential papers on deep stereo matching published in top-tier computer vision conferences and prestigious journals.

## Table of Contents

1. [Survey & Fundamentals](#fundamentals)
2. [CodeBase](#codebase)
3. [Datasets](#datasets)
   - [Real-World](#real-world)
   - [Synthetic](#synthetic) 
4. [Frameworks](#frameworks)
   - [Learning for Stereo Pipeline](#learning-for-stereo-pipeline)
      - [Matching Cost](#matching-cost)
      - [Optimization](#optimization) 
      - [Refinement](#refinement) 
   - [End-to-End Architectures](#end-to-end-architectures)
      - [Foundational Deep Stereo Architectures](#foundational)
      - [Efficient-Oriented Deep Stereo Architectures](#efficient-oriented) 
      - [Multi-Task Deep Stereo Architectures](#multi-task) 
      - [Beyond Visual Spectrum Deep Stereo Architectures](#multi-modal) 
   - [Challenges & Solutions](#challenges-and-solutions)
      - [Addressing the Over-Smoothing Issue](#over-smoothing)
      - [Missing Ground Truth Depth](#missing-gt) 
      - [Domain Shift](#domain-shift) 
      - [Transparent and Reflective (ToM) Surfaces ](#tom) 
      - [Asymmetric Stereo ](#asymmetric) 
   - [Confidence Estimation](#confidence-estimation)
5. [Workshops](#workshops)
6. [Tutorials & Talks](#tutorials-talks)



<h2 id="fundamentals"> Survey & Fundamentals </h2>

<details open><summary style="font-size: larger; font-weight: bold;"> Basics</summary><ul>

   * *"A taxonomy and evaluation of dense two-frame stereo correspondence algorithms"*, *International Journal of Computer Vision (TPAMI), 2002*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9233988&casa_token=thq8xzMfDVQAAAAA:LqT40M8CQY9Xt8j8pKTUJr2E89KAB9c1DGG1Pw9q1YMG__o5htMzH1Xx3_wlPwLcesYHgvc&tag=1)] [[Bibtex](./bibliography/taxonomy.txt)]

   * *"Evaluation of cost functions for stereo matching"*, CVPR, 2007. [[Paper](https://elib.dlr.de/53001/1/Hirschm%C3%BCllerStereoMatchingCvpr07.pdf)] [[Bibtex](./bibliography/cost-functions.txt)]


   * *"Stereo Vision: Algorithms and Applications"* [[Slides](http://vision.deis.unibo.it/~smatt/Seminars/StereoVision.pdf)] [[Bibtex](./bibliography/stereo-vision.txt)]


</details>

<details open><summary style="font-size: larger; font-weight: bold;"> Deep Stereo Matching</summary><ul>

   * *"A survey on deep learning techniques for stereo-based depth estimation"*, *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9233988&casa_token=thq8xzMfDVQAAAAA:LqT40M8CQY9Xt8j8pKTUJr2E89KAB9c1DGG1Pw9q1YMG__o5htMzH1Xx3_wlPwLcesYHgvc&tag=1)] [[Bibtex](./bibliography/survey-stereo-2.txt)]


   * *"On the synergies between machine learning and binocular stereo for depth estimation from images: a survey"*, *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2021*. [[Paper](https://arxiv.org/pdf/2004.08566.pdf)] [[Bibtex](./bibliography/OnTheSynergies.txt)]

</ul>
</details>

<details open><summary style="font-size: larger; font-weight: bold;"> Learned Confidence Estimation </summary><ul>

   * *"Quantitative evaluation of confidence measures in a machine learning world"*, *ICCV, 2017*. [[Paper](https://arxiv.org/abs/2101.00431)] [[Bibtex](./bibliography/QuantitativeConf.txt)]

   * *"On the Confidence of Stereo Matching in a Deep-Learning Era: A Quantitative Evaluation"*, *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2022*. [[Paper](https://arxiv.org/abs/2101.00431)] [[Bibtex](./bibliography/OnTheConfidence.txt)]

</ul>
</details>


## CodeBase

* **OpenStereo**: *"OpenStereo: A Comprehensive Benchmark for Stereo Matching and Strong Baseline"*, *arXiv, 2023* [[Code](https://github.com/XiandaGuo/OpenStereo)] [[Paper](https://arxiv.org/pdf/2312.00343.pdf)] [[Bibtex](./bibliography/OpenStereo.txt)]



## Datasets


<details open id="real-world"><summary style="font-size: larger; font-weight: bold;">Real-World</summary><ul>



   <details open><summary style="font-size: larger; font-weight: bold;"> RGB </summary>
   
   * **Middlebury v3**: *"High-resolution stereo datasets with subpixel-accurate ground truth"*, *GCPR 2014*. [[Paper](https://elib.dlr.de/90624/1/ScharsteinEtal2014.pdf)] [[Dataset](https://vision.middlebury.edu/stereo/eval3/)] [[Bibtex](./bibliography/Middlebury_v3.txt)]

   * **KITTI 2012**: *"Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite"*, *CVPR, 2012*. [[Paper](https://projet.liris.cnrs.fr/imagine/pub/proceedings/CVPR2012/data/papers/424_O3C-04.pdf)] [[Dataset](https://www.cvlibs.net/datasets/kitti/)] [[Bibtex](./bibliography/KITTI_2012.txt)]

   * **KITTI 2015**: *"Object Scene Flow for Autonomous Vehicles"*, *CVPR, 2015*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Menze_Object_Scene_Flow_2015_CVPR_paper.pdf)] [[Dataset](https://www.cvlibs.net/datasets/kitti/)] [[Bibtex](./bibliography/KITTI_2015.txt)]

   * **ETH3D**: *"A multi-view stereo benchmark with high-resolution images and multi-camera videos"*, *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Schops_A_Multi-View_Stereo_CVPR_2017_paper.pdf)] [[Dataset](https://www.eth3d.net/)] [[Bibtex](./bibliography/ETH3D.txt)]

   * **DrivingStereo**: *"DrivingStereo: A Large-Scale Dataset for Stereo Matching in Autonomous Driving Scenarios"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_DrivingStereo_A_Large-Scale_Dataset_for_Stereo_Matching_in_Autonomous_Driving_CVPR_2019_paper.pdf)] [[Dataset](https://drivingstereo-dataset.github.io/)] [[Bibtex](./bibliography/DrivingStereo.txt)]

   * **WSVD**: *"Web stereo video supervision for depth prediction from dynamic scenes"*, *3DV, 2019*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8885937)] [[Dataset](https://sites.google.com/view/wsvd/home)] [[Bibtex](./bibliography/WSVD.txt)]

   * **Flickr1024**: *"Flickr1024: A large-scale dataset for stereo image super-resolution"*, *ICCVW, 2019*. [[Paper](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LCI/Wang_Flickr1024_A_Large-Scale_Dataset_for_Stereo_Image_Super-Resolution_ICCVW_2019_paper.pdf)] [[Dataset](https://yingqianwang.github.io/Flickr1024/)] [[Bibtex](./bibliography/Flickr1024.txt)]

   * **Middlebury 2021 Mobile Dataset**: [[Dataset](https://vision.middlebury.edu/stereo/data/scenes2021/)] [[Bibtex](./bibliography/Middlebury_v3.txt)]

   * **The Booster Dataset**: *"Open Challenges in Deep Stereo: The Booster Dataset"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ramirez_Open_Challenges_in_Deep_Stereo_The_Booster_Dataset_CVPR_2022_paper.pdf)] [[Dataset](https://cvlab-unibo.github.io/booster-web/)] [[Bibtex](./bibliography/Booster.txt)]

   * **Holopix50k**: *"Holopix50k: A Large-Scale In-the-Wild Stereo Image Dataset"*, *CVPR, 2020*. [[Paper](https://arxiv.org/abs/2003.11172)] [[Dataset](https://leiainc.github.io/holopix50k/)]  [[Bibtex](./bibliography/Holopix50k.txt)]

   * **InStereo2K**: *"InStereo2K: A Large Real Dataset for Stereo Matching in Indoor Scenes"*, *Science China Information Sciences, 2020*. [[Paper](https://link.springer.com/article/10.1007/s11432-019-2803-x)] [[Github](https://github.com/YuhuaXu/StereoDataset)]

      
   </details>



   <details open><summary style="font-size: larger; font-weight: bold;"> Multimodal </summary>

   * **CATS**: *"CATS: A Color and Thermal Stereo Benchmark"*, *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Treible_CATS_A_Color_CVPR_2017_paper.pdf)] [[Dataset](https://bigdatavision.org/CAT/download.html)] [[Bibtex](./bibliography/CATS.txt)]

   * **RGB-NIR-Stereo**: *"Deep material-aware cross-spectral stereo matching"*, *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhi_Deep_Material-Aware_Cross-Spectral_CVPR_2018_paper.pdf)] [[Code](https://github.com/tiancheng-zhi/cs-stereo)] [[Bibtex](./bibliography/CS-Stereo.txt)]

   * **MVSEC**: *"The Multivehicle Stereo Event Camera Dataset: An Event Camera Dataset for 3D Perception"*, *RAL 2018*. [[Paper](https://arxiv.org/abs/1801.10202)] [[Dataset](https://daniilidis-group.github.io/mvsec/)] [[Bibtex](./bibliography/MVSEC.txt)]

   * **DSEC**: *"DSEC: A Stereo Event Camera Dataset for Driving Scenarios"*, *RAL, 2021*. [[Paper](https://rpg.ifi.uzh.ch/docs/RAL21_DSEC.pdf)] [[Code](https://github.com/uzh-rpg/DSEC)] [[Dataset](https://dsec.ifi.uzh.ch/)] [[Bibtex](./bibliography/DSEC.txt)]

   * **RGB-MS**: *"RGB-Multispectral Matching: Dataset, Learning Methodology, Evaluation"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tosi_RGB-Multispectral_Matching_Dataset_Learning_Methodology_Evaluation_CVPR_2022_paper.pdf)] [[Dataset](https://cvlab-unibo.github.io/rgb-ms-web/)] [[Bibtex](./bibliography/RGB-MS.txt)]
   
   * **M3ED**: *"M3ed: Multi-robot, multi-sensor, multi-environment event dataset"*, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/papers/Chaney_M3ED_Multi-Robot_Multi-Sensor_Multi-Environment_Event_Dataset_CVPRW_2023_paper.pdf)] [[Dataset](https://m3ed.io/)] [[Bibtex](./bibliography/M3ED.txt)]

   * **Gated Stereo**: *"Gated Stereo: Joint Depth Estimation from Gated and Wide-Baseline Active Stereo Cues"*, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Walz_Gated_Stereo_Joint_Depth_Estimation_From_Gated_and_Wide-Baseline_Active_CVPR_2023_paper.pdf)] [[Dataset](https://light.princeton.edu/gatedstereo/)] [[Bibtex](./bibliography/Gated.txt)]

   * **MS^2**: *"Deep Depth Estimation From Thermal Image"*, *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Shin_Deep_Depth_Estimation_From_Thermal_Image_CVPR_2023_paper.pdf)] [[Dataset](https://sites.google.com/view/multi-spectral-stereo-dataset)] [[Bibtex](./bibliography/MS2.txt)] 

   </details>




   <details open><summary style="font-size: larger; font-weight: bold;"> Rendered </summary>

   * **The NeRF-Stereo Dataset**: *"NeRF-Supervised Deep Stereo"*, *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tosi_NeRF-Supervised_Deep_Stereo_CVPR_2023_paper.pdf)] [[Dataset](https://amsacta.unibo.it/id/eprint/7218/)] [[Bibtex](./bibliography/NS-Stereo.txt)] 

   </details>

</ul>
</details>



<details open id="synthetic">
<summary style="font-size: larger; font-weight: bold;">Synthetic</summary>


* **MPI Sintel**: *"A naturalistic open source movie for optical flow evaluation"*, *ECCV, 2012*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Mehl_Spring_A_High-Resolution_High-Detail_Dataset_and_Benchmark_for_Scene_Flow_CVPR_2023_paper.pdf)] [[Dataset](http://sintel.is.tue.mpg.de/)] [[Bibtex](./bibliography/MPISintel.txt)]

* **Freiburg SceneFlow**: *"A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation"*, *CVPR, 2016*. [[Paper](https://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16/paper-MIFDB16.pdf)] [[Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)] [[Bibtex](./bibliography/SceneFlow.txt)]

* **HS-VS**: *"Hierarchical deep stereo matching on high-resolution image"*, *CVPR, 2019*. [[Paper](https://arxiv.org/pdf/1912.06704.pdf)] [[Dataset](https://github.com/gengshan-y/high-res-stereo?tab=readme-ov-file)] [[Bibtex](./bibliography/HSMNet.txt)]

* **Virtual KITTI**: *"Virtual kitti 2"*, *arXiv, 2020*. [[Paper](https://arxiv.org/pdf/2001.10773.pdf)] [[Dataset](https://europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds/)] [[Bibtex](./bibliography/VirtualKITTI.txt)]


* **TartanAir**: *"TartanAir: A dataset to push the limits of visual slam"*, *IROS, 2020*. [[Paper](https://ieeexplore.ieee.org/document/9341801?denied=)] [[Dataset](https://theairlab.org/tartanair-dataset)] [[Bibtex](./bibliography/TartanAir.txt)]

* **Semi-synthesis**: *"Semi-synthesis: A fast way to produce effective datasets for stereo matching"*, *ICCVW, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021W/WAD/papers/He_Semi-Synthesis_A_Fast_Way_To_Produce_Effective_Datasets_for_Stereo_CVPRW_2021_paper.pdf)] [[Bibtex](./bibliography/Semi-synthesis.txt)]

* **UnrealStereo4K**: *"SMD-Nets: Stereo Mixture Density Networks"*, *CVPR, 2021*. [[Paper](http://www.cvlibs.net/publications/Tosi2021CVPR.pdf)] [[Dataset](https://github.com/fabiotosi92/SMD-Nets)] [[Bibtex](./bibliography/SMD-Nets.txt)]

* **IRS**: *"IRS: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation"*, *ICME, 2021*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9428423&casa_token=wdItdcnJ3QkAAAAA:OZDqHpMb0LlKp_t_unpp8aZWh0dvRai0KkZAtdI_jvioFMqjv3goGrR3AIjmObzMjRmE_7k)] [[Dataset](https://github.com/HKBU-HPML/IRS)] [[Bibtex](./bibliography/Irs.txt)]

* **CREStereo**: *"Practical stereo matching via cascaded recurrent network with adaptive correlation"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Practical_Stereo_Matching_via_Cascaded_Recurrent_Network_With_Adaptive_Correlation_CVPR_2022_paper.pdf)] [[Dataset](https://github.com/megvii-research/CREStereo)] [[Bibtex](./bibliography/CreStereo.txt)]

* **SimStereo**: *"Active-Passive SimStereo – Benchmarking the Cross-Generalization Capabilities of Deep Learning-based Stereo Methods"*, *NeurIPS, 2022*. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/bc3a68a20e5c8ba5cbefc1ecf74bfaaa-Paper-Datasets_and_Benchmarks.pdf)] [[Dataset](https://ieee-dataport.org/open-access/active-passive-simstereo)] [[Bibtex](./bibliography/SimStereo.txt)]


* **Spring**: *"Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo"*, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Mehl_Spring_A_High-Resolution_High-Detail_Dataset_and_Benchmark_for_Scene_Flow_CVPR_2023_paper.html)] [[Dataset](http://spring-benchmark.org)] [[Bibtex](./bibliography/Spring.txt)]

* **Dynamic Replica**: *"DynamicStereo: Consistent Dynamic Depth From Stereo Videos"*, *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Karaev_DynamicStereo_Consistent_Dynamic_Depth_From_Stereo_Videos_CVPR_2023_paper.pdf)] [[Dataset](https://dynamic-stereo.github.io/)] [[Bibtex](./bibliography/Dynamic_Replica.txt)]

</details>

## Frameworks

### Learning for Stereo Pipeline

<details open id="matching-cost">
<summary style="font-size: larger; font-weight: bold;">Matching Cost</summary>


* **Deep Embed**: *"A deep visual correspondence embedding model for stereo matching costs"*, *ICCV, 2015*. [[Paper](https://openaccess.thecvf.com/content_iccv_2015/papers/Chen_A_Deep_Visual_ICCV_2015_paper.pdf)] [[Bibtex](./bibliography/Deep_Embed.txt)]

* **MC-CNN**: *"Stereo matching by training a convolutional neural network to compare image patches"*, *JMLR, 2016*. [[Paper](https://www.jmlr.org/papers/v17/15-535.html)] [[Code](https://github.com/jzbontar/mc-cnn)] [[Bibtex](./bibliography/MC-CNN.txt)]

* **Content CNN**: *"Efficient deep learning for stereo matching"*, *CVPR, 2016*. [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Luo_Efficient_Deep_Learning_CVPR_2016_paper.pdf)] [[Code](https://github.com/datvuthanh/Stereo-Matching)] [[Bibtex](./bibliography/Content_CNN.txt)]

* **Per-pixel pyramid-pooling**: *"Look wider to match image patches with convolutional neural networks"*, *SPR, 2016*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7778222)] [[Bibtex](./bibliography/per-pixel_pyramid-pooling.txt)]

* **Consistency and Distinctiveness**: *"Fundamental principles on learning new features for effective dense matching"*, *TIP, 2017*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8038003)] [[Bibtex](./bibliography/Consistency_and_Distinctiveness.txt)]

* **MC-CNN-WS**: *"Weakly supervised learning of deep metrics for stereo reconstruction"*, *ICCV, 2017*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tulyakov_Weakly_Supervised_Learning_ICCV_2017_paper.pdf)] [[Code](https://github.com/tlkvstepan/mc-cnn-ws)] [[Bibtex](./bibliography/MC-CNN-WS.txt)]

* **CBMV**: *"CBMV: A coalesced bidirectional matching volume for disparity estimation"*, *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Batsos_CBMV_A_Coalesced_CVPR_2018_paper.pdf)] [[Bibtex](./bibliography/CBMV.txt)]

* **SDC**: *"SDC - stacked dilated convolution: A unified descriptor network for dense matching tasks"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Schuster_SDC_-_Stacked_Dilated_Convolution_A_Unified_Descriptor_Network_for_CVPR_2019_paper.pdf)] [[Bibtex](./bibliography/SDC.txt)]

* **Semi-dense Stereo**: *"Semi-dense Stereo Matching using Dual CNNs"*, *WACV, 2019*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8659297)] [[Bibtex](./bibliography/Semi-dense_Stereo.txt)]

</details>

<details open id="optimization">
<summary style="font-size: larger; font-weight: bold;">Optimization</summary>

* **GCP**: *"Learning to detect ground control points for improving the accuracy of stereo matching"*, *CVPR, 2014*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2014/papers/Spyropoulos_Learning_to_Detect_2014_CVPR_paper.pdf)] [[Bibtex](./bibliography/GCP.txt)]

* **LevStereo**: *"Leveraging stereo matching with learning-based confidence measures"*, *CVPR, 2015*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Park_Leveraging_Stereo_Matching_2015_CVPR_paper.pdf)] [[Bibtex](./bibliography/LevStereo.txt)]

* **O1**: *"Learning a general-purpose confidence measure based on o (1) features and a smarter aggregation strategy for semi global matching"*, *3DV, 2016*. [[Paper](http://vision.disi.unibo.it/~mpoggi/papers/3dv2016_o1.pdf)] [[Bibtex](./bibliography/O1.txt)]

* **PBCP**: *"Patch Based Confidence Prediction for Dense Disparity Map"*, *BMVC, 2016*. [[Paper](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Seki2016BMVC.pdf)] [[Bibtex](./bibliography/PBCP.txt)]

* **Sgm-Nets**: *"Sgm-Nets: Semi-global matching with neural networks"*, *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Seki_SGM-Nets_Semi-Global_Matching_CVPR_2017_paper.pdf)] [[Bibtex](./bibliography/Sgm-Nets.txt)]

* **SGM-Forest**: *"Learning to fuse proposals from multiple scanline optimizations in semi-global matching"*, *ECCV, 2018*. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Johannes_Schoenberger_Learning_to_Fuse_ECCV_2018_paper.pdf)] [[Bibtex](./bibliography/SGM-Forest.txt)]

</details>

<details  open id="refinement">
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


### End-to-End Architectures

<details open id="foundational">
<summary style="font-size: larger; font-weight: bold;">Foundational Deep Stereo Architectures</summary><ul>


  <details open>
    <summary style="font-size: larger; font-weight: bold;">CNN-based Cost Volume Aggregation</summary><ul>


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

   * **AANet**: *"AANet: Adaptive Aggregation Network for Efficient Stereo Matching"*, *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_AANet_Adaptive_Aggregation_Network_for_Efficient_Stereo_Matching_CVPR_2020_paper.pdf)] [[Code](https://github.com/haofeixu/aanet)] [[Bibtex](./bibliography/AANet.txt)]

   * **Bi3D**: *"Bi3D: Stereo Depth Estimation via Binary Classifications"*, *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Badki_Bi3D_Stereo_Depth_Estimation_via_Binary_Classifications_CVPR_2020_paper.pdf)] [[Code](https://github.com/NVlabs/Bi3D)] [[Bibtex](./bibliography/Bi3D.txt)]

  </details>


  <details open class="nested-details">
    <summary style="font-size: larger; font-weight: bold;">3D Architectures</summary>

   * **GC-Net**: *"End-to-end learning of geometry and context for deep stereo regression"*, *ICCV, 2017*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Kendall_End-To-End_Learning_of_ICCV_2017_paper.pdf)] [[Bibtex](./bibliography/GC-Net.txt)]

   * **ECA**: *"Deep stereo matching with explicit cost aggregation sub-architecture"*, *AAAI, 2018*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/12267/12126)] [[Bibtex](./bibliography/ECA.txt)]

   * **PSMNet**: *"Pyramid Stereo Matching Network"*, *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chang_Pyramid_Stereo_Matching_CVPR_2018_paper.pdf)] [[Code](https://github.com/JiaRenChang/PSMNet)] [[Bibtex](./bibliography/PSMNet.txt)]

   * **HSMNet**: *"Hierarchical deep stereo matching on high-resolution images"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Hierarchical_Deep_Stereo_Matching_on_High-Resolution_Images_CVPR_2019_paper.pdf)] [[Code](https://github.com/gengshan-y/high-res-stereo)]  [[Bibtex](./bibliography/HSMNet.txt)]

   * **GWCNet**: *"Group-wise correlation stereo network"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_Group-Wise_Correlation_Stereo_Network_CVPR_2019_paper.pdf)] [[Code](https://github.com/xy-guo/GwcNet)]  [[Bibtex](./bibliography/GWCNet.txt)]

   * **EMCUA**: *"Multi-Level Context Ultra-Aggregation for Stereo Matching"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Nie_Multi-Level_Context_Ultra-Aggregation_for_Stereo_Matching_CVPR_2019_paper.pdf)] [[Bibtex](./bibliography/EMCUA.txt)]

   * **CSPN**: *"Learning depth with convolutional spatial propagation network"*, *TPAMI, 2019*. [[Paper](https://arxiv.org/pdf/1810.02695.pdf)] [[Code](https://github.com/XinJCheng/CSPN)] [[Bibtex](./bibliography/CSPN.txt)]

   * **GA-Net**: *"Ga-net: Guided aggregation net for end-to-end stereo matching"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_GA-Net_Guided_Aggregation_Net_for_End-To-End_Stereo_Matching_CVPR_2019_paper.pdf)] [[Code](https://github.com/feihuzhang/GANet)] [[Bibtex](./bibliography/GANet.txt)]

   * **Stereodrnet**: *"Stereodrnet: Dilated residual stereonet"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chabra_StereoDRNet_Dilated_Residual_StereoNet_CVPR_2019_paper.pdf)] [[Bibtex](./bibliography/Stereodrnet.txt)]

   * **PDSNet**: *"Practical deep stereo (pds): Toward applications-friendly deep stereo matching"*, *NeurIPS, 2018*. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/ade55409d1224074754035a5a937d2e0-Paper.pdf)] [[Code](https://github.com/tlkvstepan/PracticalDeepStereo_NIPS2018)] [[Bibtex](./bibliography/PDSNet.txt)]

   * **CasStereo**: *"Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching"*, *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gu_Cascade_Cost_Volume_for_High-Resolution_Multi-View_Stereo_and_Stereo_Matching_CVPR_2020_paper.pdf)] [[Code](https://github.com/alibaba/cascade-stereo)]  [[Bibtex](./bibliography/CasStereo.txt)]

   * **WaveletStereo**: *"WaveletStereo: Learning Wavelet Coefficients of Disparity Map in Stereo Matching"*, *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_WaveletStereo_Learning_Wavelet_Coefficients_of_Disparity_Map_in_Stereo_Matching_CVPR_2020)] [[Bibtex](./bibliography/WaveletStereo.txt)]

   * **CFNet**: *"CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching"*, *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Shen_CFNet_Cascade_and_Fused_Cost_Volume_for_Robust_Stereo_Matching_CVPR_2021_paper.pdf)] [[Code](https://github.com/gallenszl/CFNet)] [[Bibtex](./bibliography/CFNet.txt)]

   * **UASNet**: *"UASNet: Uncertainty Adaptive Sampling Network for Deep Stereo Matching"*, *ICCV, 2021* [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Mao_UASNet_Uncertainty_Adaptive_Sampling_Network_for_Deep_Stereo_Matching_ICCV_2021_paper.pdf)] [[Bibtex](./bibliography/UASNet.txt)]

   * **PCWNet**: *"PCW-Net: Pyramid Combination and Warping Cost Volume for Stereo Matching"*, *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920280.pdf)] [[Code](https://github.com/gallenszl/PCWNet)] [[Bibtex](./bibliography/PCWNet.txt)]

   * **ICVP**: *"Image-Coupled Volume Propagation for Stereo Matching"*, *ICIP, 2023*. [[Paper](https://ieeexplore.ieee.org/abstract/document/10222247)] [[Code](https://github.com/ohkwon718/icvp)] [[Bibtex](./bibliography/ICVP.txt)]

   * **SEDNet**: *"Learning the distribution of errors in stereo matching for joint disparity and uncertainty estimation"*, CVPR, 2023. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Learning_the_Distribution_of_Errors_in_Stereo_Matching_for_Joint_CVPR_2023_paper.pdf)] [[Code](https://github.com/lly00412/SEDNet)] [[Bibtex](./bibliography/SEDNet.txt)]


  </details>

  </details>

  <details open class="nested-details">
    <summary style="font-size: larger; font-weight: bold;">Neural Architecture Search (NAS)</summary>

   * **LEAStereo**: *"Hierarchical Neural Architecture Search for Deep Stereo Matching"*, *NeurIPS, 2020*. [[Paper](https://arxiv.org/pdf/2010.13501.pdf)] [[Code](https://github.com/XuelianCheng/LEAStereo)] [[Bibtex](./bibliography/LEAStereo.txt)]

   * **EASNet**: *"EASNet: searching elastic and accurate network architecture for stereo matching"*, *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920434.pdf)] [[Code](https://github.com/HKBU-HPML/EASNet)] [[Bibtex](./bibliography/EASNet.txt)]

   </details>

  <details open class="nested-details">
    <summary style="font-size: larger; font-weight: bold;">Iterative Optimized-based Architectures</summary>
    
   * **RAFT-Stereo**: *"RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching"*, *3DV, 2021*. [[Paper](https://arxiv.org/abs/2109.07547)] [[Code](https://github.com/princeton-vl/RAFT-Stereo)] [[Bibtex](./bibliography/RAFT-Stereo.txt)]

   * **SCV-Stereo**: *"SCV-Stereo: Learning Stereo Matching from a Sparse Cost Volume"*, *ICIP, 2021*. [[Paper](https://arxiv.org/abs/2107.08187)] [[Code](https://sites.google.com/view/scv-stereo)] [[Bibtex](./bibliography/SCV-Stereo.txt)] 

   * **CREStereo**: *"Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Practical_Stereo_Matching_via_Cascaded_Recurrent_Network_With_Adaptive_Correlation_CVPR_2022_paper.pdf)] [[Code](https://github.com/megvii-research/CREStereo)] [[Bibtex](./bibliography/CREStereo.txt)]

   * **EAI-Stereo**: *"EAI-Stereo: Error Aware Iterative Network for Stereo Matching"*, *ACCV, 2022*. [[Paper](https://openaccess.thecvf.com/content/ACCV2022/html/Zhao_EAI-Stereo_Error_Aware_Iterative_Network_for_Stereo_Matching_ACCV_2022_paper.html)] [[Code](https://github.com/smartadpole/EAI-Stereo)] [[Bibtex](./bibliography/EAI-Stereo.txt)]

   * **IGEV-Stereo**: *"Iterative Geometry Encoding Volume for Stereo Matching"*, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Iterative_Geometry_Encoding_Volume_for_Stereo_Matching_CVPR_2023_paper.pdf)] [[Code](https://github.com/gangweiX/IGEV)] [[Bibtex](./bibliography/IGEV-Stereo.txt)]

   * **DLNR**: *"High-Frequency Stereo Matching Network"*, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_High-Frequency_Stereo_Matching_Network_CVPR_2023_paper.pdf)] [[Code](https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network)] [[Bibtex](./bibliography/DLNR.txt)]

   * **Dynamic Stereo**: *"DynamicStereo: Consistent Dynamic Depth From Stereo Videos"*, *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Karaev_DynamicStereo_Consistent_Dynamic_Depth_From_Stereo_Videos_CVPR_2023_paper.pdf)] [[Code](https://dynamic-stereo.github.io/)] [[Bibtex](./bibliography/Dynamic_Replica.txt)]

   * **CREStereo++**: *"Uncertainty Guided Adaptive Warping for Robust and Efficient Stereo Matching"*, *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Jing_Uncertainty_Guided_Adaptive_Warping_for_Robust_and_Efficient_Stereo_Matching_ICCV_2023_paper.pdf)] [[Bibtex](./bibliography/CREStereo++.txt)]

   * **Selective-Stereo**: *"Selective-Stereo: Adaptive Frequency Information Selection for Stereo Matching"*, *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2403.00486.pdf)] [[Code](https://github.com/Windsrain/Selective-Stereo)] [[Bibtex](./bibliography/Selective-Stereo.txt)]

   * **Any-Stereo**: *"Any-Stereo: Arbitrary Scale Disparity Estimation for Iterative Stereo Matching"*, *AAAI, 2024*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28119/28242)] [[Code](https://github.com/Zhaohuai-L/Any-Stereo)] [[Bibtex](./bibliography/Any-Stereo.txt)]

   * **MC-Stereo**: *"MC-Stereo: Multi-peak Lookup and Cascade Search Range for Stereo Matching"*, *3DV, 2024*. [[Paper](https://arxiv.org/pdf/2311.02340.pdf)] [[Code](https://github.com/MiaoJieF/MC-Stereo)] [[Bibtex](./bibliography/MC-Stereo.txt)]

   * **ICGNet**: *"Learning Intra-view and Cross-view Geometric Knowledge for Stereo Matching"*, *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2402.19270.pdf)] [[Code](https://github.com/DFSDDDDD1199/ICGNet)] [[Bibtex](./bibliography/ICGNet.txt)]

   * **XR-Stereo**: *"Stereo Matching in Time: 100+ FPS Video Stereo Matching for Extended Reality"*, *WACV, 2024*. [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Cheng_Stereo_Matching_in_Time_100_FPS_Video_Stereo_Matching_for_WACV_2024_paper.pdf)] [[Code](https://github.com/za-cheng/XR-Stereo)] [[Bibtex](./bibliography/XR-Stereo.txt)]

  </details>


  <details open class="nested-details">
    <summary style="font-size: larger; font-weight: bold;">Transformer-based Architectures</summary>
    
   * **STTR**: *"Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective With Transformers"*, *ICCV, 2021*  [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Revisiting_Stereo_Depth_Estimation_From_a_Sequence-to-Sequence_Perspective_With_Transformers_ICCV_2021_paper.pdf)] [[Code](https://github.com/mli0603/stereo-transformer)] [[Bibtex](./bibliography/STTR.txt)]

   * **CEST**: *"Context-enhanced stereo transformer"*, *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920263.pdf)] [[Code](https://github.com/guoweiyu/Context-Enhanced-Stereo-Transformer)] [[Bibtex](./bibliography/CEST.txt)]

   * **Chitransformer**: *"Chitransformer: Towards Reliable Stereo From Cues"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Su_Chitransformer_Towards_Reliable_Stereo_From_Cues_CVPR_2022_paper.pdf)] [[Code](https://github.com/ISL-CV/ChiTransformer)] [[Bibtex](./bibliography/Chitransformer.txt)]

   * **GMStereo**: *"Unifying Flow, Stereo and Depth Estimation"*, *TPAMI, 2023*. [[Paper](https://arxiv.org/pdf/2211.05783.pdf)] [[Code](https://haofeixu.github.io/unimatch/)] [[Bibtex](./bibliography/GMStereo.txt)]

   * **CroCo v2**: *"CroCo v2: Improved Cross-View Completion Pre-training for Stereo Matching and Optical Flow"*, *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Weinzaepfel_CroCo_v2_Improved_Cross-view_Completion_Pre-training_for_Stereo_Matching_and_ICCV_2023_paper.pdf)] [[Code](https://github.com/naver/croco)] [[Bibtex](./bibliography/CroCo.txt)]

   * **ELFNet**: *"Elfnet: Evidential local-global fusion for stereo matching"*, *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lou_ELFNet_Evidential_Local-global_Fusion_for_Stereo_Matching_ICCV_2023_paper.pdf)] [[Code](https://openaccess.thecvf.com/content/ICCV2023/papers/Lou_ELFNet_Evidential_Local-global_Fusion_for_Stereo_Matching_ICCV_2023_paper.pdf)] [[Bibtex](./bibliography/ELFNet.txt)]

   * **GOAT**: *"Global Occlusion-Aware Transformer for Robust Stereo Matching"*, *WACV, 2024*. [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Global_Occlusion-Aware_Transformer_for_Robust_Stereo_Matching_WACV_2024_paper.pdf)] [[Code](https://github.com/Magicboomliu/GOAT)] [[Bibtex](./bibliography/GOAT.txt)]

  </details>

  <details open class="nested-details">
    <summary style="font-size: larger; font-weight: bold;">Markov Random Field-based Architectures</summary>

   * **NMRF**: *"Neural Markov Random Field for Stereo Matching"*, *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2403.11193.pdf)] [[Code](https://github.com/aeolusguan/NMRF)] [[Bibtex](./bibliography/NMRF.txt)]

  </details>

</ul>
</details>

<details open id="efficient-oriented">
<summary style="font-size: larger; font-weight: bold;">Efficient-Oriented Deep Stereo Architectures</summary>

* **Stereonet**: *"Stereonet: Guided hierarchical refinement for real-time edge-aware depth prediction"*, *ECCV, 2018*. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sameh_Khamis_StereoNet_Guided_Hierarchical_ECCV_2018_paper.pdf)] [[Code](https://github.com/neka-nat/StereoNet)] [[Bibtex](./bibliography/Stereonet.txt)] 

* **MadNet**: *"Real-Time Self-Adaptive Deep Stereo"*, *CVPR, 2019*. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tonioni_Real-Time_Self-Adaptive_Deep_Stereo_CVPR_2019_paper.pdf)] [[Code](https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo?tab=readme-ov-file)] [[Bibtex](./bibliography/MadNet.txt)]

* **Deeppruner**: *"Deeppruner: Learning efficient stereo matching via differentiable patchmatch"*, *ICCV, 2019*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Duggal_DeepPruner_Learning_Efficient_Stereo_Matching_via_Differentiable_PatchMatch_ICCV_2019_paper.pdf)] [[Code](https://github.com/uber-research/DeepPruner)] [[Bibtex](./bibliography/Deeppruner.txt)] 

* **Anytime**: *"Anytime stereo image depth estimation on mobile devices"*, *ICRA, 2019*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8794003)] [[Code](https://github.com/mileyan/AnyNet)] [[Bibtex](./bibliography/Anytime.txt)] 

* **Fast DS-CS**: *"Fast Deep Stereo with 2D Convolutional Processing of Cost Signatures"*, *WACV, 2020* [[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Yee_Fast_Deep_Stereo_with_2D_Convolutional_Processing_of_Cost_Signatures_WACV_2020_paper.pdf)]  [[Code](https://github.com/ayanc/fdscs)]  [[Bibtex](./bibliography/FDCSC.txt)]

* **AAFS**: *"Attention-Aware Feature Aggregation for Real-time Stereo Matching on Edge Devices"*, *ACCV, 2020* [[Code](https://github.com/JiaRenChang/RealtimeStereo)] [[Paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Chang_Attention-Aware_Feature_Aggregation_for_Real-time_Stereo_Matching_on_Edge_Devices_ACCV_2020_paper.pdf)] [[Bibtex](./bibliography/AAFS.txt)]

* **MABNet**: *"MABNet: a lightweight stereo network based on multibranch adjustable bottleneck module"*, *ECCV, 2020*. [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730341.pdf)] [[Code](https://github.com/JumpXing/MABNet)] [[Bibtex](./bibliography/MABNet.txt)] 

* **Fadnet**: *"Fadnet: A Fast and Accurate Network for Disparity Estimation"*, *ICRA, 2020*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9197031)] [[Code](https://github.com/HKBU-HPML/FADNet)] [[Bibtex](./bibliography/Fadnet.txt)]

* **HITNet**: *"HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching"*, *CVPR, 2021*. [[Paper](https://arxiv.org/abs/2007.12140)] [[Code](https://github.com/google-research/google-research/tree/master/hitnet)] [[Bibtex](./bibliography/HITNet.txt)]

* **BGNet**: *"Bilateral Grid Learning for Stereo Matching Networks"*, *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Bilateral_Grid_Learning_for_Stereo_Matching_Networks_CVPR_2021_paper.pdf)] [[Code](https://github.com/3DCVdeveloper/BGNet)] [[Bibtex](./bibliography/BGNet.txt)]

* **DecNet**: *"A Decomposition Model for Stereo Matching"*, *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yao_A_Decomposition_Model_for_Stereo_Matching_CVPR_2021_paper.pdf)] [[Code](https://github.com/YaoChengTang/DecNet)] [[Bibtex](./bibliography/DecNet.txt)]

* **CoEX**: *"Correlate-and-Excite: Real-Time Stereo Matching via Guided Cost Volume Excitation"*, *IROS, 2021*. [[Paper](https://antabangun.github.io/projects/CoEx/)] [[Code](https://github.com/antabangun/coex)] [[Bibtex](./bibliography/CoEX.txt)]

* **RLStereo**: *"RLStereo: Real-time stereo matching based on reinforcement learning"*, *TIP, 2021*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9614986)] [[Bibtex](./bibliography/RLStereo.txt)]

* **Separable-Stereo**: *"Separable Convolutions for Optimizing 3D Stereo Networks"*, *ICIP, 2021*. [[Paper](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/kognitive-systeme/projects/deepstereovision/)] [[Code](https://github.com/cogsys-tuebingen/separable-3D-convs-for-stereo-matching)] [[Bibtex](./bibliography/Separable-Stereo.txt)]

* **MobileStereoNet**: *"MobileStereoNet: Towards Lightweight Deep Networks for Stereo Matching"*, *WACV, 2022*. [[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Shamsafar_MobileStereoNet_Towards_Lightweight_Deep_Networks_for_Stereo_Matching_WACV_2022_paper.pdf)] [[Code](https://github.com/cogsys-tuebingen/mobilestereonet)] [[Bibtex](./bibliography/MobileStereoNet.txt)]

* **ACVNet**: *"Attention Concatenation Volume for Accurate and Efficient Stereo Matching"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Attention_Concatenation_Volume_for_Accurate_and_Efficient_Stereo_Matching_CVPR_2022_paper.pdf)] [[Code](https://github.com/gangweiX/ACVNet)] [[Bibtex](./bibliography/ACVNet.txt)]

* **PBCStereo**: *"PBCStereo: A Compressed Stereo Network with Pure Binary Convolutional Operations"*, *ACCV, 2022*. [[Paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Cai_PBCStereo_A_Compressed_Stereo_Network_with_Pure_Binary_Convolutional_Operations_ACCV_2022_paper.pdf)] [[Bibtex](./bibliography/PBCStereo.txt)]

* **PCVNet**: *"Parameterized Cost Volume for Stereo Matching"*, *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zeng_Parameterized_Cost_Volume_for_Stereo_Matching_ICCV_2023_paper.pdf)] [[Code](https://github.com/jiaxiZeng/Parameterized-Cost-Volume-for-Stereo-Matching)] [[Bibtex](./bibliography/PCVNet.txt)]

* **StereoVAE**: *"StereoVAE: A lightweight stereo-matching system using embedded GPUs"*, *ICRA, 2023*. [[Paper](https://ieeexplore.ieee.org/document/10160441)] [[Bibtex](./bibliography/StereoVAE.txt)]

* **TemporalStereo**: *"TemporalStereo: Efficient Spatial-Temporal Stereo Matching Network"*, *IROS, 2023*. [[Paper](https://arxiv.org/pdf/2211.13755.pdf)] [[Code](https://github.com/youmi-zym/TemporalStereo)]  [[Bibtex](./bibliography/TemporalStereo.txt)]

* **IINet**: *"IINet: Implicit Intra-inter Information Fusion for Real-Time Stereo Matching"*, *AAAI, 2024*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28107/28218)]  [[Bibtex](./bibliography/IINet.txt)]

* **MadNet2**: *"Federated Online Adaptation for Deep Stereo"*, *CVPR, 2024*. [[Paper](https://fedstereo.github.io/)] [[Bibtex](./bibliography/FedStereo.txt)]


</details>


<details open id="multi-task">
<summary style="font-size: larger; font-weight: bold;">Multi-Task Deep Stereo Architectures</summary><ul>

<details open>
<summary style="font-size: larger; font-weight: bold;">Semantic Stereo Matching</summary>

* **Segstereo**: *"Segstereo: Exploiting semantic information for disparity estimation"*, *ECCV, 2018*. [[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Guorun_Yang_SegStereo_Exploiting_Semantic_ECCV_2018_paper.pdf)] [[Code](https://github.com/yangguorun/SegStereo)]  [[Bibtex](./bibliography/Segstereo.txt)]

* **DSNet**: *"DSNet: Joint learning for scene segmentation and disparity estimation"*, *ICRA, 2019*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8793573)] [[Bibtex](./bibliography/DSNet.txt)]

* **Dispsegnet**: *"Dispsegnet: Leveraging semantics for end-to-end learning of disparity estimation from stereo imagery"*, *RAL, 2019*. [[Paper](https://ieeexplore.ieee.org/iel7/7083369/7339444/08624344.pdf?casa_token=6L9mfcue0jkAAAAA:iMjaxtOTrLvn-9kP5g-NbVDoSAKS5M9LNoIuT3wHecgLmtjrvyhAJ0IXNU4JZjg-4XaOiAsa)] [[Bibtex](./bibliography/Dispsegnet.txt)]


* **SSPCV-Net**: *"Semantic stereo matching with pyramid cost volumes"*, *ICCV, 2019*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Semantic_Stereo_Matching_With_Pyramid_Cost_Volumes_ICCV_2019_paper.pdf)] [[Bibtex](./bibliography/SSPCV-Net.txt)]

* **RSS-Net**: *"Real-time semantic stereo matching"*, *ICRA, 2020*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9196784/)] [[Bibtex](./bibliography/RSS-Net.txt)]

* **SGNet**: *"SGNet: Semantics Guided Deep Stereo Matching"*, *ACCV, 2020*. [[Paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Chen_SGNet_Semantics_Guided_Deep_Stereo_Matching_ACCV_2020_paper.pdf)] [[Bibtex](./bibliography/SGNet.txt)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Normal-Assisted Stereo Matching</summary>

* **NA-Stereo**: *"Normal Assisted Stereo Depth Estimation"*, *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kusupati_Normal_Assisted_Stereo_Depth_Estimation_CVPR_2020_paper.pdf)] [[Code](https://github.com/udaykusupati/Normal-Assisted-Stereo)] [[Bibtex](./bibliography/NA-Stereo.txt)]


</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Stereo Matching and Optical Flow</summary>

* **BridgeDepthFlow**: *"Bridging Stereo Matching and Optical Flow via Spatiotemporal Correspondence"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lai_Bridging_Stereo_Matching_and_Optical_Flow_via_Spatiotemporal_Correspondence_CVPR_2019_paper.pdf)] [[Code](https://github.com/lelimite4444/BridgeDepthFlow)] [[Bibtex](./bibliography/BridgeDepthFlow.txt)]

* **UnOS**: *"UnOS: Unified Unsupervised Optical-Flow and Stereo-Depth Estimation by Watching Videos"*, *CVPR, 2019*. [[Paper](https://ieeexplore.ieee.org/iel7/34/9185119/08769907.pdf)] [[Code](https://github.com/baidu-research/UnDepthflow)]  [[Bibtex](./bibliography/UnOS.txt)]

* **Feature-Level Collaboration**: *"Feature-Level Collaboration: Joint Unsupervised Learning of Optical Flow,
Stereo Depth and Camera Motion"*, *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chi_Feature-Level_Collaboration_Joint_Unsupervised_Learning_of_Optical_Flow_Stereo_Depth_CVPR_2021_paper.pdf)] [[Bibtex](./bibliography/Feature-Level_Collaboration.txt)]

* **StereoFlowGAN**: *"StereoFlowGAN: Co-training for Stereo and Flow with Unsupervised Domain Adaptation"*, *BMVC, 2023*. [[Paper](https://papers.bmvc2023.org/0240.pdf)] [[Bibtex](./bibliography/StereoFlowGAN.txt)]

</details>



<details open>
<summary style="font-size: larger; font-weight: bold;"> Scene Flow </summary>

* **FlowNet3.0**: *"Occlusions, motion and depth boundaries with a generic network for disparity, optical flow or scene flow estimation"*, *ECCV, 2018*. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Eddy_Ilg_Occlusions_Motion_and_ECCV_2018_paper.pdf)] [[Code](https://github.com/lmb-freiburg/netdef_models)] [[Bibtex](./bibliography/FlowNet3.0.txt)]

* **DRISF**: *"Deep Rigid Instance Scene Flow"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ma_Deep_Rigid_Instance_Scene_Flow_CVPR_2019_paper.pdf)] [[Bibtex](./bibliography/DRISF.txt)]

* **DeblurringSF**: *"Joint stereo video deblurring, scene flow estimation and moving object segmentation"*, *TIP, 2019*. [[Paper](https://ieeexplore.ieee.org/iel7/83/4358840/08866754.pdf)] [[Bibtex](./bibliography/DeblurringSF.txt)]

* **IOSF**: *"Learning Independent Object Motion From Unlabelled Stereoscopic Videos"*, *TPAMI, 2019*. [[Paper](IOSF )] [[Bibtex](./bibliography/IOSF.txt)]

* **EPC++**: *"Every pixel counts++: Joint learning of geometry and motion with 3d holistic understanding"*, *TPAMI, 2019*. [[Paper](https://ieeexplore.ieee.org/iel7/34/9185119/08769907.pdf)] [[Code](https://github.com/chenxuluo/EPC)]  [[Bibtex](./bibliography/EPC++.txt)]

* **SENSE**: *"Sense: A shared encoder network for scene-flow estimation"*, *ICCV, 2019*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jiang_SENSE_A_Shared_Encoder_Network_for_Scene-Flow_Estimation_ICCV_2019_paper.pdf)] [[Code](https://github.com/NVlabs/SENSE)]  [[Bibtex](./bibliography/SENSE.txt)]

* **StereoExpansion**: *"Upgrading Optical Flow to 3D Scene Flow through Optical Expansion"*, *ICCV, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Upgrading_Optical_Flow_to_3D_Scene_Flow_Through_Optical_Expansion_CVPR_2020_paper.pdf)] [[Code](https://github.com/gengshan-y/expansion)]  [[Bibtex](./bibliography/StereoExpansion.txt)]


* **DWARF**: *"Learning end-to-end scene flow by distilling single tasks knowledge"*, *AAAI, 2020*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6613/6467)] [[Code](https://github.com/FilippoAleotti/DWARF-Tensorflow)]  [[Bibtex](./bibliography/DWARF.txt)]

* **SceneFlowFields++**: *"SceneFlowFields++: Multi-frame matching, visibility prediction, and robust interpolation for scene flow estimation"*, *IJCV, 2020*. [[Paper](https://arxiv.org/pdf/1902.10099.pdf)] [[Bibtex](./bibliography/SceneFlowFields++.txt)]


* **Effiscene**: *"Effiscene: Efficient per-pixel rigidity inference for unsupervised joint learning of optical flow, depth, camera pose and motion segmentation"*, *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Jiao_EffiScene_Efficient_Per-Pixel_Rigidity_Inference_for_Unsupervised_Joint_Learning_of_CVPR_2021_paper.pdf)] [[Bibtex](./bibliography/Effiscene.txt)]

* **RigidMask**: *"Learning to Segment Rigid Motions from Two Frames"*, *CVPR, 2021*. [[Paper](http://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Learning_To_Segment_Rigid_Motions_From_Two_Frames_CVPR_2021_paper.pdf)] [[Code](https://github.com/gengshan-y/rigidmask)] [[Bibtex](./bibliography/RigidMask.txt)]

* **CamLiFlow**: *"Learning optical flow and scene flow with bidirectional camera-lidar fusion"*, *TPAMI, 2023*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10310261)] [[Code](https://github.com/MCG-NJU/CamLiFlow)] [[Bibtex](./bibliography/CamLiFlow.txt)]

* **M-FUSE**: *"M-fuse: Multi-frame fusion for scene flow estimation"*, *WACV, 2023*. [[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Mehl_M-FUSE_Multi-Frame_Fusion_for_Scene_Flow_Estimation_WACV_2023_paper.pdf)] [[Code](https://github.com/cv-stuttgart/M-FUSE)] [[Bibtex](./bibliography/M-FUSE.txt)]

* **OpticalExpansion**: *"Learning Optical Expansion from Scale Matching"*, *CVPR, 2023*. [[Paper](http://openaccess.thecvf.com/content/CVPR2023/papers/Ling_Learning_Optical_Expansion_From_Scale_Matching_CVPR_2023_paper.pdf)] [[Bibtex](./bibliography/OpticalExpansion.txt)]



</details>

</details>

<details open id="multi-modal">
<summary style="font-size: larger; font-weight: bold;">Beyond Visual Spectrum Deep Stereo Architectures</summary><ul>


<details open>
<summary style="font-size: larger; font-weight: bold;">Depth-Guided Sensor Stereo Networks</summary>

* **GSD**: *"Guided stereo matching"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Poggi_Guided_Stereo_Matching_CVPR_2019_paper.pdf)] [[Code](https://github.com/mattpoggi/guided-stereo)] [[Bibtex](./bibliography/GSD.txt)]

* **LidarStereoNet**: *"Noise-Aware Unsupervised Deep Lidar-Stereo Fusion"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cheng_Noise-Aware_Unsupervised_Deep_Lidar-Stereo_Fusion_CVPR_2019_paper.pdf)] [[Code](https://github.com/XuelianCheng/LidarStereoNet)] [[Bibtex](./bibliography/LidarStereoNet.txt)]

* **Pseudo-LiDAR++**: *"Pseudo-LiDAR++: Accurate Depth for 3D Object Detection in Autonomous Driving"*, *ICLR, 2020*. [[Paper](https://arxiv.org/pdf/1906.06310.pdf)] [[Code](https://github.com/mileyan/Pseudo_Lidar_V2)] [[Bibtex](./bibliography/Pseudo-LiDAR++.txt)]

* **S<sup>3**: *"S<sup>3</sup>: Learnable sparse signal superdensity for guided depth estimation*, CVPR, 2021. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_S3_Learnable_Sparse_Signal_Superdensity_for_Guided_Depth_Estimation_CVPR_2021_paper.pdf)] [[Bibtex](./bibliography/S3.txt)]

* **LSMD-Net**: *"LSMD-Net: LiDAR-Stereo Fusion with Mixture Density Network for Depth Sensing"*, *ACCV, 2022*. [[Paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Yin_LSMD-Net_LiDAR-Stereo_Fusion_with_Mixture_Density_Network_for_Depth_Sensing_ACCV_2022_paper.pdf)] [[Bibtex](./bibliography/LSMD-Net.txt)]

* **CamLiFlow**: *"Learning optical flow and scene flow with bidirectional camera-lidar fusion"*, *TPAMI, 2023*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10310261)] [[Code](https://github.com/MCG-NJU/CamLiFlow)] [[Bibtex](./bibliography/CamLiFlow.txt)]

* **VPP**: *"Active Stereo Without Pattern Projector"*, *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Bartolomei_Active_Stereo_Without_Pattern_Projector_ICCV_2023_paper.pdf)] [[Code](https://vppstereo.github.io/)] [[Bibtex](./bibliography/VPP.txt)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Pattern Projection-Based Stereo Networks</summary>

* **ActiveStereoNet**: *"ActiveStereoNet: End-to-End Self-Supervised Learning for Active Stereo Systems"*, *ECCV, 2018*. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yinda_Zhang_Active_Stereo_Net_ECCV_2018_paper.pdf)] [[Bibtex](./bibliography/ActiveStereoNet.txt)]

* **Polka Lines**: *"Polka Lines: Learning Structured Illumination and Reconstruction for Active Stereo"*, *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Baek_Polka_Lines_Learning_Structured_Illumination_and_Reconstruction_for_Active_Stereo_CVPR_2021_paper.pdf)] [[Bibtex](./bibliography/Polka.txt)]

* **Activezero**: *"Activezero: Mixed domain learning for active stereovision with zero annotation"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_ActiveZero_Mixed_Domain_Learning_for_Active_Stereovision_With_Zero_Annotation_CVPR_2022_paper.pdf)] [[Code](https://github.com/haosulab/ActiveZero/tree/master)] [[Bibtex](./bibliography/Activezero.txt)]

* **MonoStereoFusion**: *"Depth Estimation by Combining Binocular Stereo and Monocular Structured-Light"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Depth_Estimation_by_Combining_Binocular_Stereo_and_Monocular_Structured-Light_CVPR_2022_paper.pdf)] [[Code](https://github.com/YuhuaXu/MonoStereoFusion)] [[Bibtex](./bibliography/MonoStereoFusion.txt)]


* **Activezero++**: *"Activezero++: Mixed domain learning stereo and confidence-based depth completion with zero annotation"*, *TPAMI, 2023*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10219021&casa_token=tEPEB1O5d74AAAAA:YLArSfK76hUPakJOvHaWTylzix6QgWx6MgSprE9AhUtIpp79yXFYFOXe0H9ahHEByzJsDpY)] [[Bibtex](./bibliography/Activezero++.txt)]

</details>


<details open>
<summary style="font-size: larger; font-weight: bold;">Cross-Spectral Stereo Networks</summary>

* **CS-Stereo**: *"Deep material-aware cross-spectral stereo matching"*, *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhi_Deep_Material-Aware_Cross-Spectral_CVPR_2018_paper.pdf)] [[Code](https://github.com/tiancheng-zhi/cs-stereo)] [[Bibtex](./bibliography/CS-Stereo.txt)]

* **UCSS**: *"Unsupervised cross-spectral stereo matching by learning to synthesize"*, *AAAI, 2019*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/4894/4767)] [[Code - Unofficial](https://github.com/rish-av/cross_spectral_stereo)] [[Bibtex](./bibliography/UCSS.txt)]

* **SS-MCE**: *"There and back again: Self-supervised multispectral correspondence estimation"*, *ICRA, 2021*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9561621)]  [[Bibtex](./bibliography/SS-MCE.txt)]

* **RGB-MS**: *"RGB-Multispectral matching: Dataset, learning methodology, evaluation"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tosi_RGB-Multispectral_Matching_Dataset_Learning_Methodology_Evaluation_CVPR_2022_paper.pdf)] [[Code](https://cvlab-unibo.github.io/rgb-ms-web/)] [[Bibtex](./bibliography/RGB-MS.txt)]

* **DPS-Net**: *"DPS-Net: Deep Polarimetric Stereo Depth Estimation"*, *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Tian_DPS-Net_Deep_Polarimetric_Stereo_Depth_Estimation_ICCV_2023_paper.pdf)] [[Code](https://github.com/Ethereal-Tian/DPS-Net)] [[Bibtex](./bibliography/DPS-Net.txt)]

* **CrossSP**: *"Unsupervised Cross-Spectrum Depth Estimation by Visible-Light and Thermal Cameras"*, *TITS, 2023*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10146199)] [[Code](https://github.com/whitecrow1027/CrossSP_Depth)] [[Bibtex](./bibliography/CrossSP.txt)]


</details>


<details open>
<summary style="font-size: larger; font-weight: bold;">Event Stereo Networks</summary>

* **Event-IntensityStereo**: *"Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds"*, *ICCV, 2021*. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Mostafavi_Event-Intensity_Stereo_Estimating_Depth_by_the_Best_of_Both_Worlds_ICCV_2021_paper.pdf)] [[Code](https://github.com/yonseivnl/se-cff)] [[Bibtex](./bibliography/Event-IntensityStereo.txt)]

* **SE-CFF**: *"Stereo Depth From Events Cameras: Concentrate and Focus on the Future"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Nam_Stereo_Depth_From_Events_Cameras_Concentrate_and_Focus_on_the_CVPR_2022_paper.pdf)] [[Code](https://github.com/yonseivnl/se-cff)] [[Bibtex](./bibliography/SE-CFF.txt)]

* **SCSNet**: *"Selection and Cross Similarity for Event-Image Deep Stereo"*, *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920467.pdf)] [[Code](https://github.com/Chohoonhee/SCSNet)] [[Bibtex](./bibliography/SCSNet.txt)]

* **DTC-SPADE**: *"Discrete Time Convolution for Fast Event-Based Stereo"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Discrete_Time_Convolution_for_Fast_Event-Based_Stereo_CVPR_2022_paper.pdf)] [[Bibtex](./bibliography/DTC-SPADE.txt)]

* **ADES**: *"Learning Adaptive Dense Event Stereo From the Image Domain"*, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Cho_Learning_Adaptive_Dense_Event_Stereo_From_the_Image_Domain_CVPR_2023_paper.pdf)] [[Bibtex](./bibliography/ADES.txt)]

* **SAFE**: *"Depth From Asymmetric Frame-Event Stereo: A Divide-and-Conquer Approach"*, *WACV, 2024*. [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Chen_Depth_From_Asymmetric_Frame-Event_Stereo_A_Divide-and-Conquer_Approach_WACV_2024_paper.pdf)] [[Bibtex](./bibliography/SAFE.txt)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Gated Stereo Networks</summary>

* **GatedStereo**: *"Gated Stereo: Joint Depth Estimation from Gated and Wide-Baseline Active Stereo Cues"*, *CVPR, 2023*. [[Paper](http://openaccess.thecvf.com/content/CVPR2023/papers/Walz_Gated_Stereo_Joint_Depth_Estimation_From_Gated_and_Wide-Baseline_Active_CVPR_2023_paper.pdf)] [[WebPage](https://light.princeton.edu/publication/gatedstereo/)] [[Bibtex](./bibliography/Gated.txt)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Stereo Networks with Echoes </summary>

* **StereoEchoes**: *"Stereo Depth Estimation with Echoes"*, *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870489.pdf)] [[Bibtex](./bibliography/StereoEchoes.txt)]


</details>

</ul>
</details>



### Challenges & Solutions

<details open id="over-smoothing">
<summary style="font-size: larger; font-weight: bold;">Addressing the Over-Smoothing Issue</summary>

* **SM-CDE**: *"On the over-smoothing problem of cnn based disparity estimation"*, *ICCV, 2019*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_On_the_Over-Smoothing_Problem_of_CNN_Based_Disparity_Estimation_ICCV_2019_paper.html)] [[Bibtex](./bibliography/SM-CDE.txt)] 

* **AcfNet**: *"Adaptive Unimodal Cost Volume Filtering for Deep Stereo Matching"*, *AAAI, 2020*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6991/6845)] [[Code](https://github.com/youmi-zym/AcfNet)] [[Bibtex](./bibliography/WaveletStereo.txt)]


* **CDN**: *"Wasserstein Distances for Stereo Disparity Estimation"*, *NeurIPS, 2020*. [[Paper](https://papers.nips.cc/paper/2020/file/fe7ecc4de28b2c83c016b5c6c2acd826-Paper.pdf)] [[Code](https://github.com/Div99/W-Stereo-Disp)] [[Bibtex](./bibliography/CDN.txt)] 

* **SMD-Nets**: *"SMD-Nets: Stereo Mixture Density Networks"*, *CVPR, 2021*. [[Paper](http://www.cvlibs.net/publications/Tosi2021CVPR.pdf)] [[Code](https://github.com/fabiotosi92/SMD-Nets)] [[Bibtex](./bibliography/SMD-Nets.txt)] 

* **NDR**: "*Neural disparity refinement for arbitrary resolution stereo*", *3DV, 2021*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9665913?casa_token=3rm4WpqLb_QAAAAA:5Sa0RO547j8LsaEYUeppzB33gZJg5Y3tfiPVwM9rzs9MEAuoHSta0Kdw3Cm9NrtfOOdFkIwp)] [[Website](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR.txt)] 

* **LaC**: "*Local similarity pattern and cost self-reassembling for deep stereo matching networks*", *AAAI, 2022*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20056/19815)] [[Code](https://github.com/SpadeLiu/Lac-GwcNet)] [[Bibtex](./bibliography/LaC.txt)] 

* **ADL**: "*Adaptive Multi-Modal Cross-Entropy Loss for Stereo Matching*", *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2306.15612.pdf)] [[Code](https://github.com/xxxupeng/ADL)] [[Bibtex](./bibliography/ADL.txt)] 


</details>

<details open id="missing-gt">
<summary style="font-size: larger; font-weight: bold;">Missing Ground Truth Depth</summary><ul>

<details open>
<summary style="font-size: larger; font-weight: bold;">Self-Supervised</summary>

* **MonoDepth/StereoDepth**: *"Unsupervised monocular depth estimation with left-right consistency"*, *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Godard_Unsupervised_Monocular_Depth_CVPR_2017_paper.pdf)] [[Code](https://github.com/mrharicot/monodepth)] [[Bibtex](./bibliography/MonoDepth.txt)]

* **USM**: *"Unsupervised learning of stereo matching"*, *ICCV, 2017*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhou_Unsupervised_Learning_of_ICCV_2017_paper.pdf)] [[Bibtex](./bibliography/Flow2Stereo.txt)]

* **OASM-Net**: *"Occlusion aware stereo matching via cooperative unsupervised learning"*, *ACCV, 2018*. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-20876-9_13)] [[Bibtex](./bibliography/OASM-Net.txt)]

* **UnOS**: *"UnOS: Unified Unsupervised Optical-Flow and Stereo-Depth Estimation by Watching Videos"*, *CVPR, 2019*. [[Paper](https://ieeexplore.ieee.org/iel7/34/9185119/08769907.pdf)] [[Code](https://github.com/baidu-research/UnDepthflow)]  [[Bibtex](./bibliography/UnOS.txt)]

* **BridgeDepthFlow**: *"Bridging Stereo Matching and Optical Flow via Spatiotemporal Correspondence"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lai_Bridging_Stereo_Matching_and_Optical_Flow_via_Spatiotemporal_Correspondence_CVPR_2019_paper.pdf)] [[Code](https://github.com/lelimite4444/BridgeDepthFlow)] [[Bibtex](./bibliography/BridgeDepthFlow.txt)]

* **Correspondence Consistency**: *"Unsupervised stereo matching using confidential correspondence consistency"*, *TITS, 2019*. [[Paper](https://ieeexplore.ieee.org/iel7/6979/4358928/08721668.pdf?casa_token=YfFwIZKeGJ4AAAAA:LCkTQefrcYIOljN6Yyc55dxXCUthgkrnmJLwu7gSjD_Cd65HZ_EOpGDJq49V5GylbgxtGKG4)] [[Bibtex](./bibliography/Correspondence_Consistency.txt)]

* **Flow2Stereo**: *"Flow2Stereo: Effective Self-Supervised Learning of Optical Flow and Stereo Matching"*, *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Flow2Stereo_Effective_Self-Supervised_Learning_of_Optical_Flow_and_Stereo_Matching_CVPR_2020_paper.pdf)] [[Code](https://github.com/ppliuboy/Flow2Stereo)] [[Bibtex](./bibliography/Flow2Stereo.txt)]

* **PASMNet**: *"Parallax attention for unsupervised stereo correspondence learning"*, *TPAMI, 2020*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9206116)] [[Code](https://github.com/The-Learning-And-Vision-Atelier-LAVA/PAM)] [[Bibtex](./bibliography/PASMNet.txt)]

* **MultiscopicVision**: *"Stereo matching by self-supervision of multiscopic vision"*, *IROS, 2021*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9206116)] [[WebPage](https://sites.google.com/view/multiscopic)] [[Bibtex](./bibliography/MultiscopicVision.txt)]

* **Feature-Level Collaboration**: *"Feature-Level Collaboration: Joint Unsupervised Learning of Optical Flow,
Stereo Depth and Camera Motion"*, *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chi_Feature-Level_Collaboration_Joint_Unsupervised_Learning_of_Optical_Flow_Stereo_Depth_CVPR_2021_paper.pdf)] [[Bibtex](./bibliography/Feature-Level_Collaboration.txt)]

* **Occlusion-Aware Stereo**: *"Unsupervised Occlusion-Aware Stereo Matching With Directed Disparity Smoothing"*, *TITS, 2022*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9404889)] [[Bibtex](./bibliography/Occlusion-Aware_Stereo.txt)]

</details>
<details open >
<summary style="font-size: larger; font-weight: bold;">Cross-Framework/Proxy Supervision</summary>

* **Reversing-Stereo**: *"Reversing the cycle: self-supervised deep stereo through enhanced monocular distillation"*, *ECCV, 2020*. [[Paper](https://arxiv.org/pdf/2008.07130.pdf)] [[Code](https://github.com/FilippoAleotti/Reversing)] [[Bibtex](./bibliography/Reversing-Stereo.txt)]

* **Revealing-Stereo**: *"Revealing the Reciprocal Relations between Self-Supervised Stereo and Monocular Depth Estimation"*, *ICCV, 2021*. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Revealing_the_Reciprocal_Relations_Between_Self-Supervised_Stereo_and_Monocular_Depth_ICCV_2021_paper.pdf)]  [[Bibtex](./bibliography/Revealing-Stereo.txt)]

* **Two-in-One**: *"Two-in-one depth: Bridging the gap between monocular and binocular self-supervised depth estimation"*, *ICCV, 2023*. [[Paper](http://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Two-in-One_Depth_Bridging_the_Gap_Between_Monocular_and_Binocular_Self-Supervised_ICCV_2023_paper.pdf)] [[Code](https://github.com/ZM-Zhou/TiO-Depth_pytorch)] [[Bibtex](./bibliography/Two-in-One.txt)]

* **NeRF-Supervised Stereo**: "*NeRF-Supervised Deep Stereo*", *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tosi_NeRF-Supervised_Deep_Stereo_CVPR_2023_paper.pdf)] [[Website](https://nerfstereo.github.io/)] [[Code](https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo)] [[Bibtex](./bibliography/NS-Stereo.txt)] 

</details>




</ul>
</details>

<details open id="domain-shift">
<summary style="font-size: larger; font-weight: bold;">Domain Shift</summary><ul>

<details open>
<summary style="font-size: larger; font-weight: bold;">Zero-shot Generalization</summary><ul>

<details open>
<summary style="font-size: larger; font-weight: bold;">Domain-Agnostic Feature Modeling</summary>

* **DSM-Net**: "*Domain-invariant Stereo Matching Networks*", *ECCV, 2020*. [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470409.pdf)] [[Code](https://github.com/feihuzhang/DSMNet)] [[Bibtex](./bibliography/DSM-Net.txt)] 

* **FCStereo**: "*Revisiting Domain Generalized Stereo Matching Networks From a Feature Consistency Perspective*", *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Revisiting_Domain_Generalized_Stereo_Matching_Networks_From_a_Feature_Consistency_CVPR_2022_paper.pdf)] [[Code](https://github.com/jiaw-z/FCStereo)] [[Bibtex](./bibliography/FCStereo.txt)] 

* **GraftNet**: "*GraftNet: Towards Domain Generalized Stereo Matching With a Broad-Spectrum and Task-Oriented Feature*", *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_GraftNet_Towards_Domain_Generalized_Stereo_Matching_With_a_Broad-Spectrum_and_CVPR_2022_paper.pdf)] [[Code](https://github.com/SpadeLiu/Graft-PSMNet)] [[Bibtex](./bibliography/GraftNet.txt)] 

* **ITSA**: "*ITSA: An Information-Theoretic Approach to Automatic Shortcut Avoidance and Domain Generalization in Stereo Matching Networks*", *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chuah_ITSA_An_Information-Theoretic_Approach_to_Automatic_Shortcut_Avoidance_and_Domain_CVPR_2022_paper.pdf)] [[Code](https://github.com/waychin-weiqin/ITSA)] [[Bibtex](./bibliography/ITSA.txt)] 

* **HVT**: "*Domain Generalized Stereo Matching via Hierarchical Visual Transformation*", *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chang_Domain_Generalized_Stereo_Matching_via_Hierarchical_Visual_Transformation_CVPR_2023_paper.pdf)] [[Code](https://github.com/cty8998/HVT-PSMNet)] [[Bibtex](./bibliography/HVT.txt)] 

* **MRL-Stereo**: "*Masked representation learning for domain generalized stereo matching*", *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Rao_Masked_Representation_Learning_for_Domain_Generalized_Stereo_Matching_CVPR_2023_paper.pdf)] [[Bibtex](./bibliography/MRL-Stereo.txt)] 

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Non-parametric Cost Volumes</summary>

* **MS-Nets**: "*Matching-space Stereo Networks for Cross-domain Generalization*", *3DV, 2020*. [[Paper](https://mordohai.github.io/public/Cai_MatchingSpaceStereo20.pdf)] [[Code](https://github.com/ccj5351/MS-Nets)] [[Bibtex](./bibliography/MS-Nets.txt)] 

* **ARStereo**: "*Revisiting Non-Parametric Matching Cost Volumes for Robust and Generalizable Stereo Matching*", *NeurIPS, 2022*. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/6794f555524c9069e26970a408d353cc-Paper-Conference.pdf)] [[Code](https://github.com/kelkelcheng/AdversariallyRobustStereo)]  [[Bibtex](./bibliography/ARStereo.txt)] 

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Integration of Additional Geometric Cues</summary>

* **NDR**: "*Neural disparity refinement for arbitrary resolution stereo*", *3DV, 2021*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9665913?casa_token=3rm4WpqLb_QAAAAA:5Sa0RO547j8LsaEYUeppzB33gZJg5Y3tfiPVwM9rzs9MEAuoHSta0Kdw3Cm9NrtfOOdFkIwp)] [[Website](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR.txt)] 

* **EVHS**: "*Expansion of Visual Hints for Improved Generalization in Stereo Matching*", *WACV, 2023*. [[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Pilzer_Expansion_of_Visual_Hints_for_Improved_Generalization_in_Stereo_Matching_WACV_2023_paper.pdf)] [[Bibtex](./bibliography/EVHS.txt)] 

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Synthetic Data Generation and Domain Translation</summary>

* **StereoGAN**: "*StereoGAN: Bridging Synthetic-to-Real Domain Gap by Joint Optimization of Domain Translation and Stereo Matching*", *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_StereoGAN_Bridging_Synthetic-to-Real_Domain_Gap_by_Joint_Optimization_of_Domain_CVPR_2020_paper.pdf)] [[Code](https://github.com/ruiliu-ai/StereoGAN)] [[Bibtex](./bibliography/StereoGAN.txt)] 
 
* **LSSI**: "*Learning Stereo from Single Images*", *ECCV, 2020*. [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460698.pdf)] [[Code](https://github.com/nianticlabs/stereo-from-mono/)] [[Bibtex](./bibliography/LSSI.txt)] 

* **FoggyStereo**: "*FoggyStereo: Stereo Matching with Fog Volume Representation*", *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yao_FoggyStereo_Stereo_Matching_With_Fog_Volume_Representation_CVPR_2022_paper.pdf)] [[Code](https://github.com/nianticlabs/stereo-from-mono/)] [[Bibtex](./bibliography/FoggyStereo.txt)] 


* **NeRF-Supervised Stereo**: "*NeRF-Supervised Deep Stereo*", *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tosi_NeRF-Supervised_Deep_Stereo_CVPR_2023_paper.pdf)] [[Website](https://nerfstereo.github.io/)] [[Code](https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo)] [[Bibtex](./bibliography/NS-Stereo.txt)] 

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Knowledge Transfer</summary>

* **DKT-Stereo**: "*Robust Synthetic-to-Real Transfer for Stereo Matching*", *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2403.07705)] [[Code](https://github.com/jiaw-z/DKT-Stereo)] [[Bibtex](./bibliography/DKT-Stereo.txt)] 

</details>

</ul>
</details>


<details open>
<summary style="font-size: larger; font-weight: bold;">Offline Adaptation</summary>


* **Open-World Stereo**: *"Open-world stereo video matching with deep rnn"*, *ECCV, 2018*. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yiran_Zhong_Open-World_Stereo_Video_ECCV_2018_paper.pdf)] [[Bibtex](./bibliography/Open-World.txt)] 

* **Confidence-guided Adaptation**: *"Unsupervised adaptation for deep stereo"*, *ICCV, 2017*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tonioni_Unsupervised_Adaptation_for_ICCV_2017_paper.pdf)] [[Code](https://github.com/CVLAB-Unibo/Unsupervised-Adaptation-for-Deep-Stereo)] [[Bibtex1](./bibliography/Confidence_guided_Adaptation_0.txt)]  [[Bibtex2](./bibliography/Confidence_guided_Adaptation_1.txt)]

* **ZOLE**: *"Zoom and Learn: Generalizing Deep Stereo Matching to Novel Domain"*, *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Pang_Zoom_and_Learn_CVPR_2018_paper.pdf)] [[Code](https://github.com/jiahaopang/zole)] [[Bibtex](./bibliography/v.txt)]

* **AdaStereo**: *"AdaStereo: A Simple and Efficient Approach for Adaptive Stereo Matching"*, *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Song_AdaStereo_A_Simple_and_Efficient_Approach_for_Adaptive_Stereo_Matching_CVPR_2021_paper.pdf)] [[Bibtex](./bibliography/AdaStereo.txt)]

* **UnDAF**: *"UnDAF: A General Unsupervised Domain Adaptation Framework for Disparity or Optical Flow Estimation"*, *CVPR, 2021*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9811811)] [[Code](https://sites.google.com/view/undaf)] [[Bibtex](./bibliography/UnDAF.txt)]

* **UCFNet**: *"Digging Into Uncertainty-Based Pseudo-Label for Robust Stereo Matching"*, *TPAMI, 2023*. [[Paper](https://arxiv.org/pdf/2307.16509.pdf)] [[Code](https://github.com/gallenszl/UCFNet?tab=readme-ov-file)] [[Bibtex](./bibliography/UCFNet.txt)]

* **StereoFlowGAN**: *"StereoFlowGAN: Co-training for Stereo and Flow with Unsupervised Domain Adaptation"*, *BMVC, 2023*. [[Paper](https://papers.bmvc2023.org/0240.pdf)] [[Bibtex](./bibliography/StereoFlowGAN.txt)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Online Continual Adaptation</summary>


* **MadNet**: *"Real-Time Self-Adaptive Deep Stereo"*, *CVPR, 2019*. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tonioni_Real-Time_Self-Adaptive_Deep_Stereo_CVPR_2019_paper.pdf)] [[Code](https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo?tab=readme-ov-file)] [[Bibtex](./bibliography/MadNet.txt)]

* **Learning2Adapt**: *"Learning to adapt for stereo"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tonioni_Learning_to_Adapt_for_Stereo_CVPR_2019_paper.pdf)] [[Code](https://github.com/CVLAB-Unibo/Learning2AdaptForStereo)] [[Bibtex](./bibliography/Learning2Adapt.txt)]

* **Continual Adaptation for Deep Stereo**: *"Continual adaptation for deep stereo"*, *TPAMI, 2021*. [[Paper](https://ieeexplore.ieee.org/document/9418523?denied=)] [[Code](https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo?tab=readme-ov-file)] [[Bibtex](./bibliography/ContinualAdaptation.txt)]

* **RAG**: *"Continual Stereo Matching of Continuous Driving Scenes With Growing Architecture"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Continual_Stereo_Matching_of_Continuous_Driving_Scenes_With_Growing_Architecture_CVPR_2022_paper.pdf)] [[Bibtex](./bibliography/RAG.txt)]

* **PointFix**: *"PointFix: Learning to Fix Domain Bias for Robust Online Stereo Adaptation"*, *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136980557.pdf)] [[Bibtex](./bibliography/PointFix.txt)]

* **FedStereo**: *"Federated Online Adaptation for Deep Stereo"*, *CVPR, 2024*. [[Paper](https://fedstereo.github.io/)] [[Bibtex](./bibliography/FedStereo.txt)]

* **RAG-Continual**: *"Reusable Architecture Growth for Continual Stereo Matching"*, *TPAMI, 2024*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10475553)]  [[Code](https://github.com/chzhang18/RAG)] [[Bibtex](./bibliography/RAG-Continual.txt)]

</details>

</details>

<details open id="tom">
<summary style="font-size: larger; font-weight: bold;">Transparent and Reflective (ToM) Surfaces </summary>

* **DDF**: *"Deep Depth Fusion for Black, Transparent, Reflective and Texture-Less Objects"*, *ICRA, 2020*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9196894)]  [[Code](https://github.com/chzhang18/RAG)] [[Bibtex](./bibliography/DDF.txt)]

* **TA-Stereo**: *"Transparent Objects: A Corner Case in Stereo Matching"*, *ICRA, 2023*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10161385)]  [[Code](mias.group/TA-Stereo)] [[Bibtex](./bibliography/TA-Stereo.txt)]

* **Depth4ToM**: *"Learning Depth Estimation for Transparent and Mirror Surfaces"*, *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Costanzino_Learning_Depth_Estimation_for_Transparent_and_Mirror_Surfaces_ICCV_2023_paper.pdf)]  [[Code](https://cvlab-unibo.github.io/Depth4ToM/)] [[Bibtex](./bibliography/Depth4ToM.txt)]

</details>


<details open id="asymmetric">
<summary style="font-size: larger; font-weight: bold;">Asymmetric Stereo </summary>

* **Visually-Imbalanced Stereo**: "*Visually Imbalanced Stereo Matching*", *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Visually_Imbalanced_Stereo_Matching_CVPR_2020_paper.pdf)] [[Code](https://github.com/DandilionLau/Visually-Imbalanced-Stereo)] [[Bibtex](./bibliography/VI-SM.txt)] 

* **NDR**: "*Neural disparity refinement for arbitrary resolution stereo*", *3DV, 2021*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9665913?casa_token=3rm4WpqLb_QAAAAA:5Sa0RO547j8LsaEYUeppzB33gZJg5Y3tfiPVwM9rzs9MEAuoHSta0Kdw3Cm9NrtfOOdFkIwp)] [[Code](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR.txt)] 

* **DA-AS**: *"Degradation-agnostic Correspondence from Resolution-asymmetric Stereo"*, *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Degradation-Agnostic_Correspondence_From_Resolution-Asymmetric_Stereo_CVPR_2022_paper.pdf)] [[Bibtex](./bibliography/DA-AS.txt)]

* **SASS**: *"Unsupervised Deep Asymmetric Stereo Matching with Spatially-Adaptive Self-Similarity"*, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Song_Unsupervised_Deep_Asymmetric_Stereo_Matching_With_Spatially-Adaptive_Self-Similarity_CVPR_2023_paper.pdf)] [[Bibtex](./bibliography/SASS.txt)]



</details>


</ul>
</details>




### Confidence Estimation

<details open>
<summary style="font-size: larger; font-weight: bold;">Machine Learning Approaches</summary><ul>

<details open>
<summary style="font-size: larger; font-weight: bold;">Disparity-based</summary>

* **ENS7**: *"Ensemble learning for confidence measures in stereo vision"*, CVPR, 2013. [[Paper](http://openaccess.thecvf.com/content_cvpr_2013/papers/Haeusler_Ensemble_Learning_for_2013_CVPR_paper.pdf)]  [[Bibtex](./bibliography/ENS23.txt)]

* **O1**: *"Learning a general-purpose confidence measure based on o (1) features and a smarter aggregation strategy for semi global matching"*, 3DV, 2016. [[Paper](http://vision.disi.unibo.it/~mpoggi/papers/3dv2016_o1.pdf)]  [[Bibtex1](./bibliography/O1.txt)] [[Bibtex2](./bibliography/O1_1.txt)]


</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Cost Volume-based</summary>

* **ENS23**: *"Ensemble learning for confidence measures in stereo vision"*, CVPR, 2013. [[Paper](http://openaccess.thecvf.com/content_cvpr_2013/papers/Haeusler_Ensemble_Learning_for_2013_CVPR_paper.pdf)]  [[Bibtex](./bibliography/ENS23.txt)]

* **GCP**: *"Learning to detect ground control points for improving the accuracy of stereo matching"*, CVPR, 2014. [[Paper](https://openaccess.thecvf.com/content_cvpr_2014/papers/Spyropoulos_Learning_to_Detect_2014_CVPR_paper.pdf)]  [[Bibtex1](./bibliography/GCP.txt)] [[Bibtex2](./bibliography/GCP1.txt)]

* **LEV**: *"Leveraging stereo matching with learning-based confidence measures"*, CVPR, 2015. [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Park_Leveraging_Stereo_Matching_2015_CVPR_paper.pdf)]  [[Bibtex1](./bibliography/LEV.txt)] [[Bibtex2](./bibliography/LEV1.txt)]

* **FA**: *"Feature augmentation for learning confidence measure in stereo matching"*, TIP, 2017. [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Park_Leveraging_Stereo_Matching_2015_CVPR_paper.pdf)]  [[Bibtex](./bibliography/FA.txt)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">SGM-specific</summary>

* **SGMForest**: *"Learning to fuse proposals from multiple scanline optimizations in semi-global matching"*, ECCV, 2018. [[Paper](http://vision.disi.unibo.it/~mpoggi/papers/3dv2016_o1.pdf)]  [[Bibtex](./bibliography/SGMForest.txt)]


</details>

</ul>
</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Deep Learning Approaches</summary><ul>

<details open>
<summary style="font-size: larger; font-weight: bold;">Disparity-based</summary>

* **CCNN**: *"Learning from scratch a confidence measure"*, BMVC, 2016. [[Paper](https://www.researchgate.net/profile/Stefano-Mattoccia-2/publication/317191595_Learning_from_scratch_a_confidence_measure/links/593f9db3a6fdcc1b10aac9ec/Learning-from-scratch-a-confidence-measure.pdf)] [[Code](https://github.com/fabiotosi92/CCNN-Tensorflow)] [[Bibtex](./bibliography/CCNN.txt)]

* **PBCP**: *"Patch Based Confidence Prediction for Dense Disparity Map"*, BMVC, 2016. [[Paper](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Seki2016BMVC.pdf)]  [[Bibtex](./bibliography/PBCP.txt)]

* **EFN/LFN**: *"Stereo matching confidence learning based on multi-modal convolution neural networks"*, RFMI, 2017. [[Paper](http://www.arts-pi.org.tn/rfmi2017/papers/10_CameraReadySubmission_llncs2e%20(3).pdf)] [[Bibtex](./bibliography/EFN.txt)]

* **MMC**: *"Learning confidence measures by multi-modal convolutional neural networks"*, WACV, 2018. [[Paper](WACV)] [[Bibtex](./bibliography/MMC.txt)]

* **LGC/ConfNet**: *"Beyond local reasoning for stereo confidence estimation with deep learning"*, ECCV, 2018. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Fabio_Tosi_Beyond_local_reasoning_ECCV_2018_paper.pdf)] [[Code](https://github.com/fabiotosi92/LGC-Tensorflow)] [[Bibtex](./bibliography/LGC.txt)]

* **SEDNet**: *"Learning the distribution of errors in stereo matching for joint disparity and uncertainty estimation"*, CVPR, 2023. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Learning_the_Distribution_of_Errors_in_Stereo_Matching_for_Joint_CVPR_2023_paper.pdf)] [[Code](https://github.com/lly00412/SEDNet)] [[Bibtex](./bibliography/SEDNet.txt)]




</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Cost Volume-based</summary>

* **RCN**: *"Improved stereo matching with constant highway networks and reflective confidence learning"*, CVPR, 2017. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Shaked_Improved_Stereo_Matching_CVPR_2017_paper.pdf)] [[Code](https://github.com/amitshaked/resmatch)] [[Bibtex](./bibliography/RCN.txt)]

* **MPN**: *"Deep stereo confidence prediction for depth estimation"*, ICIP, 2017. [[Paper](https://ieeexplore.ieee.org/iel7/8267582/8296222/08296430.pdf)] [[Bibtex](./bibliography/MPN.txt)]

* **UCN**: *"Unified confidence estimation networks for robust stereo matching"*, TIP, 2018. [[Paper](https://ieeexplore.ieee.org/iel7/83/4358840/08510870.pdf)] [[Bibtex](./bibliography/UCN.txt)]

* **LAF**: *"Laf-net: Locally adaptive fusion networks for stereo confidence estimation"*, CVPR, 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_LAF-Net_Locally_Adaptive_Fusion_Networks_for_Stereo_Confidence_Estimation_CVPR_2019_paper.pdf)] [[Bibtex](./bibliography/LAF.txt)]

* **CRNN**: *"Pixel-Wise Confidences for Stereo Disparities Using Recurrent Neural Networks"*, BMVC, 2019. [[Paper](https://publica.fraunhofer.de/bitstreams/c1f200e0-49e6-488c-84de-d217a550bdf6/download)] [[Bibtex](./bibliography/CRNN.txt)]

* **ACN**: *"Adversarial confidence estimation networks for robust stereo matching"*, TITS, 2020. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_LAF-Net_Locally_Adaptive_Fusion_Networks_for_Stereo_Confidence_Estimation_CVPR_2019_paper.pdf)] [[Bibtex](./bibliography/ACN.txt)]

* **CVA**: *"Cnn-based cost volume analysis as confidence measure for dense matching"*, ICCVW, 2019. [[Paper](http://openaccess.thecvf.com/content_ICCVW_2019/papers/3DRW/Mehltretter_CNN-Based_Cost_Volume_Analysis_as_Confidence_Measure_for_Dense_Matching_ICCVW_2019_paper.pdf)] [[Bibtex](./bibliography/CVA.txt)]


</details>

</ul>
</details>


## Workshops

* **NTIRE 2023: HR Depth from Images of Specular and Transparent Surfaces**. P. Z. Ramirez, F. Tosi, L. Di Stefano, R. Timofte
A. Costanzino, M. Poggi, S. Salti, S. Mattoccia; CVPRW 2023, Vancouver, Canada [[Website](https://cvlab-unibo.github.io/booster-web/ntire.html)]

* **Robust Vision Challenge (ROB)**, Zendel et al., ECCV 2022 [[Website](http://www.robustvision.net/)]




<h2 id="tutorials-talks"> Tutorials & Talks </h2>

* **Facing depth estimation in-the-wild with deep networks**. M. Poggi, F. Tosi, F. Aleotti, K. Batsos, P. Mordohai, S. Mattoccia; ECCV 2020, SEC, Glasgow [[Website](https://sites.google.com/view/eccv-2020-robust-depth/home)]

* **Learning and understanding single image depth estimation in the wild**.  M. Poggi, F. Tosi, F. Aleotti, S. Mattoccia, C. Godard, J. Watson, M. Firman, G.J. Brostow; CVPR 2020, Seattle, Washington, US  [[Website](https://sites.google.com/view/cvpr-2020-depth-from-mono/home)]

* **Learning-based depth estimation from stereo and monocular images: successes, limitations and future challenges**. M. Poggi, F. Tosi, K. Batsos, P. Mordohai, S. Mattoccia, CVPR 2019, Long Beach, California, US [[Website](https://sites.google.com/view/cvpr-2019-depth-from-image/home)]

* **Learning-based depth estimation from stereo and monocular images: successes, limitations and future challenges**.  M. Poggi, F. Tosi, K. Batsos, P. Mordohai, S. Mattoccia; 3DV 2018, Verona, Italy [[Website](https://sites.google.com/view/3dv-2018-depth-from-image/home)]

* **Lecture: Computer Vision (Prof. Andreas Geiger, University of Tübingen)**. [[Preliminaries](https://www.youtube.com/watch?v=6hr6xpOU-uw&list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_&index=12&pp=iAQB)] [[Block Matching](https://www.youtube.com/watch?v=EVzEJQl8WFk&list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_&index=13&t=830s&pp=iAQB)] [[Siamese Networks](https://www.youtube.com/watch?v=vLgsiIXNf0I&list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_&index=14&pp=iAQB)] [[Spatial Regularization](https://www.youtube.com/watch?v=gqz6R1qChVQ&list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_&index=15&t=359s&pp=iAQB)] [[End-to-End Learning](https://www.youtube.com/watch?v=9vrmwZ9Pl4o&list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_&index=16&t=639s&pp=iAQB)] 


