# Awesome-Deep-Stereo-Matching [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
Welcome to the "Awesome-Deep-Stereo-Matching" repository, a curated list of state-of-the-art deep stereo matching resources maintained by [Fabio Tosi](https://fabiotosi92.github.io/), [Matteo Poggi](https://mattpoggi.github.io/) and [Luca Bartolomei](https://www.unibo.it/sitoweb/luca.bartolomei5), from the University of Bologna. This repository, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision), aims to provide a comprehensive collection of the latest and most influential papers on deep stereo matching published in top-tier computer vision conferences and prestigious journals.

The methods included in this repository are appropriately categorized to facilitate navigation and understanding of the diverse approaches and techniques employed in deep stereo matching research. Additionally, for anyone in need, we also release the [reference bib](https://github.com/fabiotosi92/Awesome-Deep-Stereo-Matching/blob/main/references.bib) which contains the bib entries for all the works included in this page.

We use the :triangular_flag_on_post: symbol to highlight the absolute most groundbreaking works.

:rotating_light: :rotating_light: :rotating_light: This repository is closely associated with our surveys on deep stereo matching: 

1) ["A Survey on Deep Stereo Matching in the Twenties"](https://link.springer.com/content/pdf/10.1007/s11263-024-02331-0.pdf), Tosi et al., IJCV 2025
2) ["On the Synergies between Machine Learning and Binocular Stereo for Depth Estimation from Images: a Survey"](https://arxiv.org/abs/2004.08566), Poggi et al., TPAMI 2021
3) ["On the confidence of stereo matching in a deep-learning era: a quantitative evaluation"](https://arxiv.org/abs/2101.00431), Poggi et al., TPAMI 2022

These surveys provides an in-depth overview of the field, complementing the curated list of resources found in this repository. 

Additionally, we presented a tutorial on this topic at CVPR 2024. For more information about the tutorial, including slides and additional resources, please visit our [Tutorial Webpage](https://sites.google.com/view/stereo-twenties).

If you find this repository valuable, please consider [citing](#fountain_pen-citation) it in your work and giving it a star ! :star:


**Full reference(s):**
* "A Survey on Deep Stereo Matching in the Twenties", Tosi et al., *IJCV, 2025*. [[Paper](https://link.springer.com/content/pdf/10.1007/s11263-024-02331-0.pdf)] [[Bibtex](./bibliography/inTheTwenties.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=A+Survey+on+Deep+Stereo+Matching+in+the+Twenties+&btnG=)] [[Tutorial](https://sites.google.com/view/stereo-twenties)]

* *"On the synergies between machine learning and binocular stereo for depth estimation from images: a survey"*, Poggi et al., *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2021*. [[Paper](https://arxiv.org/pdf/2004.08566.pdf)] [[Bibtex](./bibliography/OnTheSynergies.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=On+the+synergies+between+machine+learning+and+binocular+stereo+for+depth+estimation+from+images:+a+survey&btnG=)]

* *"On the Confidence of Stereo Matching in a Deep-Learning Era: A Quantitative Evaluation"*, Poggi et al., *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2022*. [[Paper](https://arxiv.org/abs/2101.00431)] [[Bibtex](./bibliography/OnTheConfidence.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=On+the+Confidence+of+Stereo+Matching+in+a+Deep-Learning+Era:+A+Quantitative+Evaluation&btnG=)]

#### [How to submit a pull request?](https://github.com/fabiotosi92/Awesome-Deep-Stereo-Matching/blob/main/how-to-PR.txt)



## :bookmark_tabs: Table of Contents

1. [Survey & Fundamentals](#fundamentals)
2. [CodeBase](#codebase)
3. [Datasets](#file_cabinet-datasets)
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
   - [Architectural Analysis](#architectural-analysis)
   - [Challenges & Solutions](#challenges-and-solutions)
      - [Addressing the Over-Smoothing Issue](#over-smoothing)
      - [Missing Ground Truth Depth](#missing-gt) 
      - [Domain Shift](#domain-shift) 
      - [Adverse Weather ](#adverse-weather) 
      - [Transparent and Reflective (ToM) Surfaces ](#tom) 
      - [Asymmetric Stereo ](#asymmetric) 
      - [Temporal Consistency ](#temporal-consistency) 
      - [Continuous Estimation Problem ](#continuous-estimation) 
   - [Confidence Estimation](#confidence-estimation)
5. [Applications](#applications)
6. [Workshops](#workshops)
7. [Tutorials & Talks](#tutorials-talks)
8. [Demos](#demos)
9. [Citation](#fountain_pen-citation)


<h2 id="fundamentals"> Surveys & Fundamentals </h2>

<details open><summary style="font-size: larger; font-weight: bold;"> Stereo Matching Basics</summary><ul>

   * *"A taxonomy and evaluation of dense two-frame stereo correspondence algorithms"*, Scharstein & Szeliski, *International Journal of Computer Vision (TPAMI), 2002*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9233988&casa_token=thq8xzMfDVQAAAAA:LqT40M8CQY9Xt8j8pKTUJr2E89KAB9c1DGG1Pw9q1YMG__o5htMzH1Xx3_wlPwLcesYHgvc&tag=1)] [[Bibtex](./bibliography/taxonomy.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+taxonomy+and+evaluation+of+dense+two-frame+stereo+correspondence+algorithms&btnG=)] 

   * *"Evaluation of cost functions for stereo matching"*, Hirschmuller & Scharstein, CVPR, 2007. [[Paper](https://elib.dlr.de/53001/1/Hirschm%C3%BCllerStereoMatchingCvpr07.pdf)] [[Bibtex](./bibliography/cost-functions.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Evaluation+of+cost+functions+for+stereo+matching&btnG=)]

   * **SGM**: *"Stereo processing by semiglobal matching and mutual information"*, Heiko Hirschmuller, TPAMI, 2007. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4359315&casa_token=NOG3dp9K27gAAAAA:IGAioT6Q0C8ofBnal-o_YIzd77YMCYRwSFRJ5YIOn58WRj1KnFayOedgkSspsw4t4Txt-kwT)] [[Bibtex](./bibliography/SGM.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Stereo+Processing+by+Semi-Global+Matching+and+Mutual+Information&btnG=)]

   * *"Computer Vision: Algorithms and Applications", 2nd Edition - (Chapter 12, Depth Estimation)*, Richard Szeliski [[Slides](http://cv2.csie.ntu.edu.tw/CV2/2023/textbook.pdf)] [[Bibtex](./bibliography/Computer-Vision.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Stereo+Vision:+Algorithms+and+Applications&btnG=)]

   * *"Stereo Matching"*, Richard Szeliski, University of Washington [[Slides](https://courses.cs.washington.edu/courses/cse576/08sp/lectures/Stereo.pdf)] 

   * *"Stereo Vision"*, Fei-Fei Li, Stanford Vision Lab [[Slides](http://vision.stanford.edu/teaching/cs131_fall1415/lectures/lecture9_10_stereo_cs131.pdf)] 

   * *"Stereo Vision: Algorithms and Applications"*, Stefano  Mattoccia, University of Bologna [[Slides](http://vision.deis.unibo.it/~smatt/Seminars/StereoVision.pdf)] [[Bibtex](./bibliography/stereo-vision.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Stereo+Vision:+Algorithms+and+Applications&btnG=)]



</details>

<details open><summary style="font-size: larger; font-weight: bold;"> Deep Stereo Matching</summary><ul>

   * *"A Survey on Deep Stereo Matching in the Twenties"*, Tosi et al., *IJCV, 2025*. [[Paper](https://link.springer.com/content/pdf/10.1007/s11263-024-02331-0.pdf)] [[Bibtex](./bibliography/inTheTwenties.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=A+Survey+on+Deep+Stereo+Matching+in+the+Twenties+&btnG=)] [[Tutorial](https://sites.google.com/view/stereo-twenties)]

   * *"A survey on deep learning techniques for stereo-based depth estimation"*, Laga et al., *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9233988&casa_token=thq8xzMfDVQAAAAA:LqT40M8CQY9Xt8j8pKTUJr2E89KAB9c1DGG1Pw9q1YMG__o5htMzH1Xx3_wlPwLcesYHgvc&tag=1)] [[Bibtex](./bibliography/survey-stereo-2.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+survey+on+deep+learning+techniques+for+stereo-based+depth+estimation&btnG=)]

   * *"On the synergies between machine learning and binocular stereo for depth estimation from images: a survey"*, Poggi et al., *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2021*. [[Paper](https://arxiv.org/pdf/2004.08566.pdf)] [[Bibtex](./bibliography/OnTheSynergies.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=On+the+synergies+between+machine+learning+and+binocular+stereo+for+depth+estimation+from+images:+a+survey&btnG=)]

</ul>
</details>

<details open><summary style="font-size: larger; font-weight: bold;"> Learned Confidence Estimation </summary><ul>

   * *"Quantitative evaluation of confidence measures in a machine learning world"*, Poggi et al., *ICCV, 2017*. [[Paper](https://arxiv.org/abs/2101.00431)] [[Bibtex](./bibliography/QuantitativeConf.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Quantitative+evaluation+of+confidence+measures+in+a+machine+learning+world&btnG=)]

   * *"On the Confidence of Stereo Matching in a Deep-Learning Era: A Quantitative Evaluation"*, Poggi et al., *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2022*. [[Paper](https://arxiv.org/abs/2101.00431)] [[Bibtex](./bibliography/OnTheConfidence.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=On+the+Confidence+of+Stereo+Matching+in+a+Deep-Learning+Era:+A+Quantitative+Evaluation&btnG=)]

</ul>

</details>


<details open><summary style="font-size: larger; font-weight: bold;"> Event-Based Stereo</summary><ul>

   * *"Event-based Stereo Depth Estimation: A Survey"*, Ghosh et al., *arXiv pre-print, 2024*. [[Paper](https://arxiv.org/pdf/2409.17680)] [[Bibtex](./bibliography/Event-Stereo-Survey.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Event-based+Stereo+Depth+Estimation%3A+A+Survey&btnG=)]

</ul>
</details>

## CodeBase

* **OpenStereo**: *"OpenStereo: A Comprehensive Benchmark for Stereo Matching and Strong Baseline"*, Xianda et al., *arXiv, 2023*  [[Paper](https://arxiv.org/pdf/2312.00343.pdf)] [[Code](https://github.com/XiandaGuo/OpenStereo)] [[Bibtex](./bibliography/OpenStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=OpenStereo:+A+Comprehensive+Benchmark+for+Stereo+Matching+and+Strong+Baseline&btnG=)]



## :file_cabinet: Datasets


<details open id="real-world"><summary style="font-size: larger; font-weight: bold;">Real-World</summary><ul>



   <details open id="real-world RGB"><summary style="font-size: larger; font-weight: bold;"> RGB </summary>
   
   * **KITTI 2012**: *"Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite"*, Geiger et al., *CVPR, 2012*. [[Paper](https://projet.liris.cnrs.fr/imagine/pub/proceedings/CVPR2012/data/papers/424_O3C-04.pdf)] [[Dataset](https://www.cvlibs.net/datasets/kitti/)] [[Bibtex](./bibliography/KITTI_2012.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Are+we+ready+for+Autonomous+Driving?+The+KITTI+Vision+Benchmark+Suite&btnG=)]

   * **KITTI 2015**: *"Object Scene Flow for Autonomous Vehicles"*, Menze et al., *CVPR, 2015*. [[Paper](http://openaccess.thecvf.com/content_cvpr_2015/papers/Menze_Object_Scene_Flow_2015_CVPR_paper.pdf)] [[Dataset](https://www.cvlibs.net/datasets/kitti/)] [[Bibtex](./bibliography/KITTI_2015.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Object+Scene+Flow+for+Autonomous+Vehicles&btnG=)]

   * **Middlebury v3**: *"High-resolution stereo datasets with subpixel-accurate ground truth"*, Scharstein et al., *GCPR 2014*. [[Paper](https://elib.dlr.de/90624/1/ScharsteinEtal2014.pdf)] [[Dataset](https://vision.middlebury.edu/stereo/eval3/)] [[Bibtex](./bibliography/Middlebury_v3.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=High-resolution+stereo+datasets+with+subpixel-accurate+ground+truth&btnG=)]

   * **Cityscapes**: *"The cityscapes dataset for semantic urban scene understanding"*, Cordts et al., *CVPR, 2016*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.pdf)] [[Dataset](https://www.cityscapes-dataset.com/)] [[Bibtex](./bibliography/Cityscapes.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=he+cityscapes+dataset+for+semantic+urban+scene+understandin&btnG=)]

   * **ETH3D**: *"A multi-view stereo benchmark with high-resolution images and multi-camera videos"*, Schops et al., *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Schops_A_Multi-View_Stereo_CVPR_2017_paper.pdf)] [[Dataset](https://www.eth3d.net/)] [[Bibtex](./bibliography/ETH3D.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+multi-view+stereo+benchmark+with+high-resolution+images+and+multi-camera+videos&btnG=)]

   * **DrivingStereo**: *"DrivingStereo: A Large-Scale Dataset for Stereo Matching in Autonomous Driving Scenarios"*, Yang et al., *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_DrivingStereo_A_Large-Scale_Dataset_for_Stereo_Matching_in_Autonomous_Driving_CVPR_2019_paper.pdf)] [[Dataset](https://drivingstereo-dataset.github.io/)] [[Bibtex](./bibliography/DrivingStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=DrivingStereo:+A+Large-Scale+Dataset+for+Stereo+Matching+in+Autonomous+Driving+Scenarios&btnG=)]

   * **WSVD**: *"Web stereo video supervision for depth prediction from dynamic scenes"*, Wang et al., *3DV, 2019*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8885937)] [[Dataset](https://sites.google.com/view/wsvd/home)] [[Bibtex](./bibliography/WSVD.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Web+stereo+video+supervision+for+depth+prediction+from+dynamic+scenes&btnG=)]

   * **Flickr1024**: *"Flickr1024: A large-scale dataset for stereo image super-resolution"*, Wang et al., *ICCVW, 2019*. [[Paper](http://openaccess.thecvf.com/content_ICCVW_2019/papers/LCI/Wang_Flickr1024_A_Large-Scale_Dataset_for_Stereo_Image_Super-Resolution_ICCVW_2019_paper.pdf)] [[Dataset](https://yingqianwang.github.io/Flickr1024/)] [[Bibtex](./bibliography/Flickr1024.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Flickr1024:+A+large-scale+dataset+for+stereo+image+super-resolution&btnG=)]

   * **ApolloScape**: *"The apolloscape open dataset for autonomous driving and its application"*, Huang et al., *TPAMI, 2019*. [[Paper](https://arxiv.org/pdf/1803.06184)] [[Dataset](https://github.com/ApolloScapeAuto/dataset-api)] [[Bibtex](./bibliography/ApolloScape.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=The+ApolloScape+Open+Dataset+for+Autonomous+Driving+and+its+Application&btnG=)]

   * **Holopix50k**: *"Holopix50k: A Large-Scale In-the-Wild Stereo Image Dataset"*, Hua et al., *CVPR, 2020*. [[Paper](https://arxiv.org/abs/2003.11172)] [[Dataset](https://leiainc.github.io/holopix50k/)]  [[Bibtex](./bibliography/Holopix50k.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Holopix50k:+A+Large-Scale+In-the-Wild+Stereo+Image+Dataset&btnG=)]

   * **A2D2**: *"Audi Autonomous Driving Dataset"*, Geyer et al., *arXiv, 2020*. [[Paper](https://arxiv.org/pdf/2004.06320.pdf)] [[Dataset](https://www.a2d2.audi/a2d2/en/dataset.html)] [[Bibtex](./bibliography/A2D2.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=A2d2%3A+Audi+autonomous+driving+dataset&btnG=)]

   * **InStereo2K**: *"InStereo2K: A Large Real Dataset for Stereo Matching in Indoor Scenes"*, Bao et al., *Science China Information Sciences, 2020*. [[Paper](https://link.springer.com/article/10.1007/s11432-019-2803-x)] [[Github](https://github.com/YuhuaXu/StereoDataset)] [[Bibtex](./bibliography/InStereo2K.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=InStereo2K%3A+A+Large+Real+Dataset+for+Stereo+Matching+in+Indoor+Scenes&btnG=)]

   * **Middlebury 2021 Mobile Dataset**: [[Dataset](https://vision.middlebury.edu/stereo/data/scenes2021/)] [[Bibtex](./bibliography/Middlebury_v3.txt)]

   * **DurLAR**: *"DurLAR: A High-Fidelity 128-Channel LiDAR Dataset with Panoramic Ambient and Reflectivity Imagery for Multi-Modal Autonomous Driving Applications"*, Li et al., *3DV, 2021*. [[Paper](https://durham-repository.worktribe.com/OutputFile/1138972)] [[Dataset](https://github.com/l1997i/DurLAR)] [[Bibtex](./bibliography/DurLAR.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=DurLAR%3A+A+High-Fidelity+128-Channel+LiDAR+Dataset+with+Panoramic+Ambient+and+Reflectivity+Imagery+for+Multi-Modal+Autonomous+Driving+Applications&btnG=)]

   * **Booster**: *"Open Challenges in Deep Stereo: The Booster Dataset"*, Ramirez et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ramirez_Open_Challenges_in_Deep_Stereo_The_Booster_Dataset_CVPR_2022_paper.pdf)] [[Dataset](https://cvlab-unibo.github.io/booster-web/WHU-stereo)] [[Bibtex](./bibliography/Booster.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Open+Challenges+in+Deep+Stereo:+The+Booster+Dataset&btnG=)]

   * **WHU-Stereo**: *"WHU-Stereo: A challenging benchmark for stereo matching of high-resolution satellite images"*, Li et al., *TGRS, 2023*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10044710&casa_token=qEuHzPZoVEoAAAAA:pU9i3aHcnnJqZuCmgqVpTEjK3UqrxBe5g48THFHdvIH7CKhP480X31SBeCJHMnVHGKkTUCE_&tag=1)] [[Dataset](https://github.com/Sheng029/WHU-Stereo)] [[Bibtex](./bibliography/WHU-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=WHU-stereo%3A+A+challenging+benchmark+for+stereo+matching+of+high-resolution+satellite+images&btnG=)]


   * **SID**: *"SID: Stereo Image Dataset for Autonomous Driving in Adverse Conditions"*, El-Shair et al., *NAECON, 2024*. [[Paper](https://arxiv.org/pdf/2407.04908)] [[Dataset](https://deepblue.lib.umich.edu/data/concern/data_sets/cc08hg37c)] [[Bibtex](./bibliography/SID.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=SID%3A+Stereo+Image+Dataset+for+Autonomous+Driving+in+Adverse+Conditions&btnG=)]


   * **LayeredFlow**: *"LayeredFlow: A Real-World Benchmark for Non-Lambertian Multi-Layer Optical Flow"*, Wen et al., *ECCV, 2024*. [[Paper](https://arxiv.org/pdf/2409.05688)] [[Dataset](https://github.com/princeton-vl/LayeredFlow)] [[Bibtex](./bibliography/LayeredFlow.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=LayeredFlow%3A+A+Real-World+Benchmark+for+Non-Lambertian+Multi-Layer+Optical+Flow&btnG=)]

   * **South Kensington SV**: *"Match Stereo Videos via Bidirectional Alignment"*, Jing et al., *arXiv, 2024*. [[Paper](https://arxiv.org/pdf/2409.20283)] [[Code&Dataset](https://github.com/TomTomTommi/bidavideo)] [[Bibtex](./bibliography/BiDA-Video.txt)] [[WebPage](https://tomtomtommi.github.io/BiDAVideo/)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Match+Stereo+Videos+via+Bidirectional+Alignment&btnG=)]

   </details>


   <details open><summary style="font-size: larger; font-weight: bold;"> Beyond-RGB </summary><ul>


   <details open><summary style="font-size: larger; font-weight: bold;"> Event Cameras</summary>

   * **UZH-RPG**: *"Semi-Dense 3D Reconstruction with a Stereo Event Camera"*, Zhou et al., *ECCV, 2018*. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yi_Zhou_Semi-Dense_3D_Reconstruction_ECCV_2018_paper.pdf)] [[Dataset](https://rpg.ifi.uzh.ch/ECCV18_stereo_davis.html)] [[Bibtex](./bibliography/UZH-RPG.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Semi-dense+3D+reconstruction+with+a+stereo+event+camera&btnG=)]

   * **MVSEC**: *"The Multivehicle Stereo Event Camera Dataset: An Event Camera Dataset for 3D Perception"*, Zhu et al., *RAL 2018*. [[Paper](https://arxiv.org/abs/1801.10202)] [[Dataset](https://daniilidis-group.github.io/mvsec/)] [[Bibtex](./bibliography/MVSEC.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=The+Multivehicle+Stereo+Event+Camera+Dataset:+An+Event+Camera+Dataset+for+3D+Perception&btnG=)]

   * **DSEC**: *"DSEC: A Stereo Event Camera Dataset for Driving Scenarios"*, Gehrig et al., *RAL, 2021*. [[Paper](https://rpg.ifi.uzh.ch/docs/RAL21_DSEC.pdf)] [[Code](https://github.com/uzh-rpg/DSEC)] [[Dataset](https://dsec.ifi.uzh.ch/)] [[Bibtex](./bibliography/DSEC.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=DSEC:+A+Stereo+Event+Camera+Dataset+for+Driving+Scenarios&btnG=)]

   * **SHEF**: *"Stereo Hybrid Event-Frame (SHEF) Cameras for 3D Perception"*, Wang et al., *IROS, 2021*. [[Paper](https://arxiv.org/pdf/2110.04988)] [[Dataset](https://github.com/ziweiWWANG/SHEF)] [[Bibtex](./bibliography/SHEF.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Stereo+hybrid+event-frame+%28shef%29+cameras+for+3d+perception&btnG=)]

   * **TUM-VIE**: *"TUM-VIE: The TUM Stereo Visual-Inertial Event Data Set"*, Klenk et al., *IROS, 2021*. [[Paper](https://arxiv.org/pdf/2108.07329)] [[Dataset](https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset)] [[Bibtex](./bibliography/TUM-VIE.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=TUM-VIE%3A+The+TUM+Stereo+Visual-Inertial+Event+Data+Set&btnG=)]
   
   * **EVIMO2**: *"EVIMO2: An Event Camera Dataset for Motion Segmentation, Optical Flow, Structure from Motion, and Visual Inertial Odometry in Indoor Scenes with Monocular or Stereo Algorithms"*, Burner et al., *ArXiv, 2022*. [[Paper](https://arxiv.org/pdf/2205.03467)] [[Dataset](https://better-flow.github.io/evimo/download_evimo_2.html)] [[Bibtex](./bibliography/EVIMO2.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=EVIMO2%3A+An+Event+Camera+Dataset+for+Motion+Segmentation%2C+Optical+Flow%2C+Structure+from+Motion%2C+and+Visual+Inertial+Odometry+in+Indoor+Scenes+with+Monocular+or+Stereo+Algorithms&btnG=)]

   * **VECtor**: *"VECtor: A Versatile Event-Centric Benchmark for Multi-Sensor SLAM"*, Gao et al., *RAL, 2022*. [[Paper](https://arxiv.org/pdf/2207.01404)] [[Dataset](https://star-datasets.github.io/vector/)] [[Bibtex](./bibliography/VECtor.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=VECtor%3A+A+Versatile+Event-Centric+Benchmark+for+Multi-Sensor+SLAM&btnG=)]

   * **ESVIO**: *"ESVIO: Event-based Stereo Visual Inertial Odometry"*, Chen et al., *RAL, 2023*. [[Paper](https://arxiv.org/pdf/2212.13184)] [[Dataset](https://github.com/arclab-hku/ESVIO)] [[Bibtex](./bibliography/ESVIO.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=ESVIO%3A+Event-based+Stereo+Visual+Inertial+Odometry&btnG=)]

   * **M3ED**: *"M3ED: Multi-Robot, Multi-Sensor, Multi-Environment Event Dataset"*, Chaney et al., *CVPRW, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023W/EventVision/papers/Chaney_M3ED_Multi-Robot_Multi-Sensor_Multi-Environment_Event_Dataset_CVPRW_2023_paper.pdf)] [[Dataset](https://m3ed.io/)] [[Bibtex](./bibliography/M3ED.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=M3ed:+Multi-robot,+multi-sensor,+multi-environment+event+dataset&btnG=)]

   </details>
   <details open><summary style="font-size: larger; font-weight: bold;"> Thermal-RGB </summary>

   * **CATS**: *"CATS: A Color and Thermal Stereo Benchmark"*, Treible et al.,  *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Treible_CATS_A_Color_CVPR_2017_paper.pdf)] [[Dataset](https://bigdatavision.org/CAT/download.html)] [[Bibtex](./bibliography/CATS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=InStereo2K:+A+Large+Real+Dataset+for+Stereo+Matching+in+Indoor+Scenes&btnG=)]

   * **MS^2**: *"Deep Depth Estimation From Thermal Image"*, Shin et al., *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Shin_Deep_Depth_Estimation_From_Thermal_Image_CVPR_2023_paper.pdf)] [[Dataset](https://sites.google.com/view/multi-spectral-stereo-dataset)] [[Bibtex](./bibliography/MS2.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Deep+Depth+Estimation+From+Thermal+Image&btnG=)]
   </details>

   <details open><summary style="font-size: larger; font-weight: bold;"> NIR-Spectral </summary>

   * **RGB-NIR-Stereo**: *"Deep material-aware cross-spectral stereo matching"*, Zhi et al., *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhi_Deep_Material-Aware_Cross-Spectral_CVPR_2018_paper.pdf)] [[Code](https://github.com/tiancheng-zhi/cs-stereo)] [[Bibtex](./bibliography/CS-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Deep+material-aware+cross-spectral+stereo+matching&btnG=)]

   * **RGB-MS**: *"RGB-Multispectral Matching: Dataset, Learning Methodology, Evaluation"*, Tosi et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tosi_RGB-Multispectral_Matching_Dataset_Learning_Methodology_Evaluation_CVPR_2022_paper.pdf)] [[Dataset](https://cvlab-unibo.github.io/rgb-ms-web/)] [[Bibtex](./bibliography/RGB-MS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=RGB-Multispectral+Matching:+Dataset,+Learning+Methodology,+Evaluation&btnG=)]
   </details>

   <details open><summary style="font-size: larger; font-weight: bold;"> Active-Gated </summary>

   * **Gated Stereo**: *"Gated Stereo: Joint Depth Estimation from Gated and Wide-Baseline Active Stereo Cues"*, Walz et al., *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Walz_Gated_Stereo_Joint_Depth_Estimation_From_Gated_and_Wide-Baseline_Active_CVPR_2023_paper.pdf)] [[Dataset](https://light.princeton.edu/gatedstereo/)] [[Bibtex](./bibliography/Gated.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Gated+Stereo:+Joint+Depth+Estimation+from+Gated+and+Wide-Baseline+Active+Stereo+Cues&btnG=)]
   </details>

   <details open><summary style="font-size: larger; font-weight: bold;"> Polarimetric </summary>

   * **RPS/IPS**: *"DPS-Net: Deep Polarimetric Stereo Depth Estimation"*, Tian et al., *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Tian_DPS-Net_Deep_Polarimetric_Stereo_Depth_Estimation_ICCV_2023_paper.pdf)] [[Dataset](https://github.com/Ethereal-Tian/DPS-Net)] [[Bibtex](./bibliography/DPS-Net.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=DPS-Net:+Deep+Polarimetric+Stereo+Depth+Estimation&btnG=)]
   </details>


   </ul>
   </details>


   <details open><summary style="font-size: larger; font-weight: bold;"> NeRF-Rendered </summary>

   * **The NeRF-Stereo Dataset**: *"NeRF-Supervised Deep Stereo"*, Tosi et al., *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tosi_NeRF-Supervised_Deep_Stereo_CVPR_2023_paper.pdf)] [[Dataset](https://amsacta.unibo.it/id/eprint/7218/)] [[Bibtex](./bibliography/NS-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=NeRF-Supervised+Deep+Stereo&btnG=)] 

   </details>

</ul>
</details>



<details open id="synthetic">
<summary style="font-size: larger; font-weight: bold;">Synthetic</summary>


* **MPI Sintel**: *"A naturalistic open source movie for optical flow evaluation"*, Butler et al., *ECCV, 2012*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Mehl_Spring_A_High-Resolution_High-Detail_Dataset_and_Benchmark_for_Scene_Flow_CVPR_2023_paper.pdf)] [[Dataset](http://sintel.is.tue.mpg.de/)] [[Bibtex](./bibliography/MPISintel.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+naturalistic+open+source+movie+for+optical+flow+evaluation&btnG=)]

* **Freiburg SceneFlow**: *"A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation"*, Mayer et al., *CVPR, 2016*. [[Paper](https://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16/paper-MIFDB16.pdf)] [[Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)] [[Bibtex](./bibliography/SceneFlow.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+Large+Dataset+to+Train+Convolutional+Networks+for+Disparity,+Optical+Flow,+and+Scene+Flow+Estimation&btnG=)]

* **Virtual KITTI**, *"Virtual worlds as proxy for multi-object tracking analysis"*, Gaidon et al., *CVPR, 2016*. [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gaidon_Virtual_Worlds_as_CVPR_2016_paper.pdf)] [[Dataset](https://europe.naverlabs.chttps://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gaidon_Virtual_Worlds_as_CVPR_2016_paper.pdfom/research-old2/computer-vision/proxy-virtual-worlds-vkitti-1/)] [[Bibtex](./bibliography/VirtualKITTI.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Virtual+worlds+as+proxy+for+multi-object+tracking+analysis&btnG=)]

* **Falling Things**: *"A synthetic dataset for 3d object detection and pose estimation"*, Tremblay et al., *CVPRW, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w40/Tremblay_Falling_Things_A_CVPR_2018_paper.pdf)] [[Dataset](https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation)] [[Bibtex](./bibliography/FallingThings.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Falling+things%3A+A+synthetic+dataset+for+3d+object+detection+and+pose+estimation&btnG=)]

* **HS-VS**: *"Hierarchical deep stereo matching on high-resolution image"*, Yang et al., *CVPR, 2019*. [[Paper](https://arxiv.org/pdf/1912.06704.pdf)] [[Dataset](https://github.com/gengshan-y/high-res-stereo?tab=readme-ov-file)] [[Bibtex](./bibliography/HSMNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Hierarchical+deep+stereo+matching+on+high-resolution+image&btnG=)]

* **Virtual KITTI 2**, Cabon et al., *arXiv, 2020*. [[Paper](https://arxiv.org/pdf/2001.10773.pdf)] [[Dataset](https://europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds/)] [[Bibtex](./bibliography/VirtualKITTI2.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Virtual+kitti+2&btnG=)]


* **TartanAir**: *"TartanAir: A dataset to push the limits of visual slam"*, Wang et al., *IROS, 2020*. [[Paper](https://ieeexplore.ieee.org/document/9341801?denied=)] [[Dataset](https://theairlab.org/tartanair-dataset)] [[Bibtex](./bibliography/TartanAir.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=TartanAir:+A+dataset+to+push+the+limits+of+visual+slam&btnG=)]

* **Semi-synthesis**: *"Semi-synthesis: A fast way to produce effective datasets for stereo matching"*, He et al., *ICCVW, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021W/WAD/papers/He_Semi-Synthesis_A_Fast_Way_To_Produce_Effective_Datasets_for_Stereo_CVPRW_2021_paper.pdf)] [[Bibtex](./bibliography/Semi-synthesis.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Semi-synthesis:+A+fast+way+to+produce+effective+datasets+for+stereo+matching&btnG=)]

* **UnrealStereo4K**: *"SMD-Nets: Stereo Mixture Density Networks"*, Tosi et al., *CVPR, 2021*. [[Paper](http://www.cvlibs.net/publications/Tosi2021CVPR.pdf)] [[Dataset](https://github.com/fabiotosi92/SMD-Nets)] [[Bibtex](./bibliography/SMD-Nets.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=SMD-Nets:+Stereo+Mixture+Density+Networks&btnG=)]

* **IRS**: *"IRS: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation"*, Wang et al., *ICME, 2021*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9428423&casa_token=wdItdcnJ3QkAAAAA:OZDqHpMb0LlKp_t_unpp8aZWh0dvRai0KkZAtdI_jvioFMqjv3goGrR3AIjmObzMjRmE_7k)] [[Dataset](https://github.com/HKBU-HPML/IRS)] [[Bibtex](./bibliography/IRS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=IRS:+A+large+naturalistic+indoor+robotics+stereo+dataset+to+train+deep+models+for+disparity+and+surface+normal+estimation&btnG=)]

* **CREStereo**: *"Practical stereo matching via cascaded recurrent network with adaptive correlation"*, Li et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Practical_Stereo_Matching_via_Cascaded_Recurrent_Network_With_Adaptive_Correlation_CVPR_2022_paper.pdf)] [[Dataset](https://github.com/megvii-research/CREStereo)] [[Bibtex](./bibliography/CREStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Practical+stereo+matching+via+cascaded+recurrent+network+with+adaptive+correlation&btnG=)]

* **SimStereo**: *"Active-Passive SimStereo – Benchmarking the Cross-Generalization Capabilities of Deep Learning-based Stereo Methods"*, Jospin et al., *NeurIPS, 2022*. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/bc3a68a20e5c8ba5cbefc1ecf74bfaaa-Paper-Datasets_and_Benchmarks.pdf)] [[Dataset](https://ieee-dataport.org/open-access/active-passive-simstereo)] [[Bibtex](./bibliography/SimStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Active-Passive+SimStereo+–+Benchmarking+the+Cross-Generalization+Capabilities+of+Deep+Learning-based+Stereo+Methods&btnG=)]

* **SimSIN**: *"Toward practical monocular indoor depth estimation"*, Wu et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Toward_Practical_Monocular_Indoor_Depth_Estimation_CVPR_2022_paper.pdf)] [[Dataset](https://github.com/facebookresearch/DistDepth)] [[Bibtex](./bibliography/SimSIN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Toward+practical+monocular+indoor+depth+estimation&btnG=)]

* **Spring**: *"Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo"*, Mehl et al., *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Mehl_Spring_A_High-Resolution_High-Detail_Dataset_and_Benchmark_for_Scene_Flow_CVPR_2023_paper.html)] [[Dataset](http://spring-benchmark.org)] [[Bibtex](./bibliography/Spring.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Spring:+A+High-Resolution+High-Detail+Dataset+and+Benchmark+for+Scene+Flow,+Optical+Flow+and+Stereo&btnG=)]

* **Dynamic Replica**: *"DynamicStereo: Consistent Dynamic Depth From Stereo Videos"*, Karaev et al., *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Karaev_DynamicStereo_Consistent_Dynamic_Depth_From_Stereo_Videos_CVPR_2023_paper.pdf)] [[Dataset](https://dynamic-stereo.github.io/)] [[Bibtex](./bibliography/Dynamic_Replica.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=DynamicStereo:+Consistent+Dynamic+Depth+From+Stereo+Videos&btnG=)]

* **All-In-One Drive**: *"A Comprehensive Perception Dataset with High-Density Long-Range Point Clouds"*, Weng et al., *arXiv  2023*. [[Paper](https://openreview.net/pdf?id=yl9aThYT9W)] [[Dataset](http://www.aiodrive.org/)] [[Bibtex](./bibliography/All-In-One.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=2005&sciodt=0%2C5&cites=13426829701666173810&scipsc=&q=All-In-One+Drive%3A+A+Comprehensive+Perception+Dataset+with+High-Density+Long-Range+Point+Clouds&btnG=)]

* **PLT-D3**: *"PLT-D3: A High-fidelity Dynamic Driving Simulation Dataset for Stereo Depth and Scene Flow"*, Tokarsky et al., *arXiv  2024*. [[Paper](https://arxiv.org/pdf/2406.07667)] [[Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/36SQKM)] [[Bibtex](./bibliography/PLT-D3.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=PLT-D3%3A+A+High-fidelity+Dynamic+Driving+Simulation+Dataset+for+Stereo+Depth+and+Scene+Flow&btnG=)]

* **QPS Dataset**: *"Disparity Estimation Using a Quad-Pixel Sensor"*, Wu et al., *BMVC, 2024*. [[Paper](https://arxiv.org/abs/2409.00665)] [[WebPage](http://www.ok.sc.e.titech.ac.jp/res/QP/index.html)] [[Dataset](https://www.dropbox.com/scl/fi/0sijedbg2kdqetmw16s9t/QP-Data-Zhuofeng-Wu.zip?rlkey=m33xsb70b4ivk0riyvg4fgk2u&st=k2fn4zkq&dl=0)] [[Bibtex](./bibliography/QPDNet.txt)]  [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Disparity+Estimation+Using+a+Quad-Pixel+Sensor&btnG=)]


* **Infinigen Stereo Video**: *"Match Stereo Videos via Bidirectional Alignment"*, Jing et al., *arXiv, 2024*. [[Paper](https://arxiv.org/pdf/2409.20283)] [[Code&Dataset](https://github.com/TomTomTommi/bidavideo)] [[Bibtex](./bibliography/BiDA-Video.txt)] [[WebPage](https://tomtomtommi.github.io/BiDAVideo/)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Match+Stereo+Videos+via+Bidirectional+Alignment&btnG=)]

</details>

## Frameworks

### Learning for Stereo Pipeline

<details open id="matching-cost">
<summary style="font-size: larger; font-weight: bold;">Matching Cost</summary>


* **Deep Embed**: *"A deep visual correspondence embedding model for stereo matching costs"*, Chen et al., *ICCV, 2015*. [[Paper](https://openaccess.thecvf.com/content_iccv_2015/papers/Chen_A_Deep_Visual_ICCV_2015_paper.pdf)] [[Bibtex](./bibliography/Deep_Embed.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+deep+visual+correspondence+embedding+model+for+stereo+matching+costs&btnG=)]

* :triangular_flag_on_post: **MC-CNN**: *"Stereo matching by training a convolutional neural network to compare image patches"*, Zbontar & LeCun, *JMLR, 2016*. [[Paper]( )] [[Code](https://github.com/jzbontar/mc-cnn)] [[Bibtex1](./bibliography/MC-CNN1.txt)]  [[Bibtex2](./bibliography/MC-CNN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Computing+the+Stereo+Matching+Cost+with+a+Convolutional+Neural+Network&btnG=)]

* **Content CNN**: *"Efficient deep learning for stereo matching"*, Luo et al., *CVPR, 2016*. [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Luo_Efficient_Deep_Learning_CVPR_2016_paper.pdf)] [[Code](https://github.com/datvuthanh/Stereo-Matching)] [[Bibtex](./bibliography/Content_CNN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Efficient+deep+learning+for+stereo+matching&btnG=)]

* **Per-pixel pyramid-pooling**: *"Look wider to match image patches with convolutional neural networks"*, Park et al., *SPR, 2016*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7778222)] [[Bibtex](./bibliography/per-pixel_pyramid-pooling.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Look+wider+to+match+image+patches+with+convolutional+neural+networks&btnG=)]

* **Consistency and Distinctiveness**: *"Fundamental principles on learning new features for effective dense matching"*, Zhang et al., *TIP, 2017*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8038003)] [[Bibtex](./bibliography/Consistency_and_Distinctiveness.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Fundamental+principles+on+learning+new+features+for+effective+dense+matching&btnG=)]

* **MC-CNN-WS**: *"Weakly supervised learning of deep metrics for stereo reconstruction"*, Tulyakov et al., *ICCV, 2017*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tulyakov_Weakly_Supervised_Learning_ICCV_2017_paper.pdf)] [[Code](https://github.com/tlkvstepan/mc-cnn-ws)] [[Bibtex](./bibliography/MC-CNN-WS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Weakly+supervised+learning+of+deep+metrics+for+stereo+reconstruction&btnG=)]

* **CBMV**: *"CBMV: A coalesced bidirectional matching volume for disparity estimation"*, Batsos et al., *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Batsos_CBMV_A_Coalesced_CVPR_2018_paper.pdf)] [[Bibtex](./bibliography/CBMV.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=CBMV:+A+coalesced+bidirectional+matching+volume+for+disparity+estimation&btnG=)]

* **SDC**: *"SDC - stacked dilated convolution: A unified descriptor network for dense matching tasks"*, Schuster et al., *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Schuster_SDC_-_Stacked_Dilated_Convolution_A_Unified_Descriptor_Network_for_CVPR_2019_paper.pdf)] [[Bibtex](./bibliography/SDC.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=SDC+-+stacked+dilated+convolution:+A+unified+descriptor+network+for+dense+matching+tasks&btnG=)]

* **Semi-dense Stereo**: *"Semi-dense Stereo Matching using Dual CNNs"*, Mao et al., *WACV, 2019*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8659297)] [[Bibtex](./bibliography/Semi-dense_Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Semi-dense+Stereo+Matching+using+Dual+CNNs&btnG=)]

</details>

<details open id="optimization">
<summary style="font-size: larger; font-weight: bold;">Optimization</summary>

* **GCP**: *"Learning to detect ground control points for improving the accuracy of stereo matching"*, Spyropoulos et al., *CVPR, 2014*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2014/papers/Spyropoulos_Learning_to_Detect_2014_CVPR_paper.pdf)] [[Bibtex](./bibliography/GCP.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+to+detect+ground+control+points+for+improving+the+accuracy+of+stereo+matching&btnG=)]

* **LevStereo**: *"Leveraging stereo matching with learning-based confidence measures"*, Park et al., *CVPR, 2015*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Park_Leveraging_Stereo_Matching_2015_CVPR_paper.pdf)] [[Bibtex](./bibliography/LevStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Leveraging+stereo+matching+with+learning-based+confidence+measures&btnG=)]

* **O1**: *"Learning a general-purpose confidence measure based on o (1) features and a smarter aggregation strategy for semi global matching"*, Poggi et al., *3DV, 2016*. [[Paper](http://vision.disi.unibo.it/~mpoggi/papers/3dv2016_o1.pdf)] [[Bibtex](./bibliography/O1.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+a+general-purpose+confidence+measure+based+on+o+(1)+features+and+a+smarter+aggregation+strategy+for+semi+global+matching&btnG=)]

* **PBCP**: *"Patch Based Confidence Prediction for Dense Disparity Map"*, Seki et al., *BMVC, 2016*. [[Paper](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Seki2016BMVC.pdf)] [[Bibtex](./bibliography/PBCP.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Patch+Based+Confidence+Prediction+for+Dense+Disparity+Map&btnG=)]

* **Sgm-Nets**: *"Sgm-Nets: Semi-global matching with neural networks"*, Seki et al., *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Seki_SGM-Nets_Semi-Global_Matching_CVPR_2017_paper.pdf)] [[Bibtex](./bibliography/Sgm-Nets.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Sgm-Nets:+Semi-global+matching+with+neural+networks&btnG=)]

* **SGM-Forest**: *"Learning to fuse proposals from multiple scanline optimizations in semi-global matching"*, Schonberger et al., *ECCV, 2018*. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Johannes_Schoenberger_Learning_to_Fuse_ECCV_2018_paper.pdf)] [[Bibtex](./bibliography/SGM-Forest.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+to+fuse+proposals+from+multiple+scanline+optimizations+in+semi-global+matching&btnG=)]

</details>

<details  open id="refinement">
<summary style="font-size: larger; font-weight: bold;">Refinement</summary>

* **RCN**: *"Improved stereo matching with constant highway networks and reflective confidence learning"*, Shaked et al., *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Shaked_Improved_Stereo_Matching_CVPR_2017_paper.pdf)] [[Code](https://github.com/amitshaked/resmatch)] [[Bibtex](./bibliography/GDN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Improved+stereo+matching+with+constant+highway+networks+and+reflective+confidence+learning&btnG=)]

* **DRR**: *"Detect, replace, refine: Deep structured prediction for pixel wise labeling"*, Gidaris et al., *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Gidaris_Detect_Replace_Refine_CVPR_2017_paper.pdf)] [[Code](https://github.com/gidariss/DRR_struct_pred/)] [[Bibtex](./bibliography/DRR.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Detect,+replace,+refine:+Deep+structured+prediction+for+pixel+wise+labeling&btnG=)]

* **OSD**: *"Efficient stereo matching leveraging deep local and context information"*, Ye et al., *IEEE Access, 2017*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8047938)] [[Bibtex](./bibliography/OSD.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Efficient+stereo+matching+leveraging+deep+local+and+context+information&btnG=)]

* **Recresnet**: *"Recresnet: A recurrent residual cnn architecture for disparity map enhancement"*, Batsos et al., *3DV, 2018*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8490974)] [[Code](https://github.com/kbatsos/RecResNet)] [[Bibtex](./bibliography/Recresnet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Recresnet:+A+recurrent+residual+cnn+architecture+for+disparity+map+enhancement&btnG=)]

* **LRCR**: *"Left-right comparative recurrent model for stereo matching"*, Jie et al., *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jie_Left-Right_Comparative_Recurrent_CVPR_2018_paper.pdf)] [[Bibtex](./bibliography/LRCR.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Left-right+comparative+recurrent+model+for+stereo+matching&btnG=)]

* **FD-Fusion**: *"Fast stereo disparity maps refinement by fusion of data-based and model-based estimations"*, Ferrera et al., *3DV, 2019*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8886031)] [[Code](https://github.com/ferreram/FD-Fusion)] [[Bibtex](./bibliography/FD-Fusion.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Fast+stereo+disparity+maps+refinement+by+fusion+of+data-based+and+model-based+estimations&btnG=)]

* **VRN**: *"Learned collaborative stereo refinement"*, Knobelreiter et al., *IJCV, 2021*. [[Paper](https://link.springer.com/article/10.1007/s11263-021-01485-5)] [[Bibtex](./bibliography/VRN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learned+collaborative+stereo+refinement&btnG=)]


* **NDR**: "*Neural disparity refinement for arbitrary resolution stereo*", Aleotti et al., *3DV, 2021*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9665913?casa_token=3rm4WpqLb_QAAAAA:5Sa0RO547j8LsaEYUeppzB33gZJg5Y3tfiPVwM9rzs9MEAuoHSta0Kdw3Cm9NrtfOOdFkIwp)] [[Website](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR.txt)] 

* **NDR v2**: "*Neural disparity refinement*", Tosi et al., *TPAMI, 2024*. [[Paper](https://ieeexplore.ieee.org/iel8/34/4359286/10552115.pdf)] [[Website](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR_v2.txt)] 

</details>


### End-to-End Architectures

<details open id="foundational">
<summary style="font-size: larger; font-weight: bold;">Foundational Deep Stereo Architectures</summary><ul>


  <details open>
    <summary style="font-size: larger; font-weight: bold;">CNN-based Cost Volume Aggregation</summary><ul>


  <details open>
    <summary style="font-size: larger; font-weight: bold;">2D Architectures</summary>
    
   * :triangular_flag_on_post: **DispNet-C**: *"A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation"*, Mayer et al.,*CVPR, 2016*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Mayer_A_Large_Dataset_CVPR_2016_paper.pdf)] [[Bibtex](./bibliography/SceneFlow.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+large+dataset+to+train+convolutional+networks+for+disparity,+optical+flow,+and+scene+flow+estimation&btnG=)]

   * **CNN+CRF**: *"End-to-end training of hybrid CNN-CRF models for stereo"*, Knobelreiter et al., *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Knobelreiter_End-To-End_Training_of_CVPR_2017_paper.pdf)]  [[Code](https://github.com/tuananh1007/End-to-End-Training-of-Hybrid-CNN-CRF-Models-for-Stereo)]  [[Bibtex](./bibliography/SceneFlow.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=End-to-end+training+of+hybrid+CNN-CRF+models+for+stereo&btnG=)]

   * **CRL**: *"Cascade residual learning: A two-stage convolutional neural network for stereo matching"*, Pang et al., *CVPRW, 2017*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Pang_Cascade_Residual_Learning_ICCV_2017_paper.pdf)]  [[Code](https://github.com/jiahaopang/crl)]  [[Bibtex](./bibliography/CRL.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Cascade+residual+learning:+A+two-stage+convolutional+neural+network+for+stereo+matching&btnG=)]

   * **iResNet**: *"Learning for disparity estimation through feature constancy"*, Liang et al., *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liang_Learning_for_Disparity_CVPR_2018_paper.pdf)]  [[Code](https://github.com/leonzfa/iResNet)]  [[Bibtex](./bibliography/iResNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+for+disparity+estimation+through+feature+constancy&btnG=)]

   * **DispNet-CSS**: *"Occlusions, motion and depth boundaries with a generic network for disparity, optical flow or scene flow estimation"*, Ilg et al., *ECCV, 2018*. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Eddy_Ilg_Occlusions_Motion_and_ECCV_2018_paper.pdf)]  [[Code](https://github.com/lmb-freiburg/netdef_models)]  [[Bibtex](./bibliography/DispNet-CSS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Occlusions,+motion+and+depth+boundaries+with+a+generic+network+for+disparity,+optical+flow+or+scene+flow+estimation&btnG=)]

   * **EdgeStereo**: *"Edgestereo: A context integrated residual pyramid network for stereo matching"*, Song et al., *ACCV, 2018*. [[Paper](https://arxiv.org/pdf/1803.05196.pdf)]   [[Bibtex](./bibliography/Edgestereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Edgestereo:+A+context+integrated+residual+pyramid+network+for+stereo+matching&btnG=)]
   
   * **AutoDispNet-CSS**: *"Autodispnet: Improving disparity estimation with automl"*, Saikia et al., *ICCV, 2019*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Saikia_AutoDispNet_Improving_Disparity_Estimation_With_AutoML_ICCV_2019_paper.pdf)]  [[Code](https://github.com/lmb-freiburg/autodispnet)]  [[Bibtex](./bibliography/AutoDispNet-CSS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Autodispnet:+Improving+disparity+estimation+with+automl&btnG=)]

   * **HD<sup>3**: *"Hierarchical discrete distribution decomposition for match density estimation"*, Yin et al., *ICCV, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yin_Hierarchical_Discrete_Distribution_Decomposition_for_Match_Density_Estimation_CVPR_2019_paper.pdf)]  [[Code](https://github.com/ucbdrive/hd3)]  [[Bibtex](./bibliography/HD3.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Hierarchical+discrete+distribution+decomposition+for+match+density+estimation&btnG=)]

   * **AANet**: *"AANet: Adaptive Aggregation Network for Efficient Stereo Matching"*, Xu et al., *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_AANet_Adaptive_Aggregation_Network_for_Efficient_Stereo_Matching_CVPR_2020_paper.pdf)] [[Code](https://github.com/haofeixu/aanet)] [[Bibtex](./bibliography/AANet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=AANet:+Adaptive+Aggregation+Network+for+Efficient+Stereo+Matching&btnG=)]

   * **Bi3D**: *"Bi3D: Stereo Depth Estimation via Binary Classifications"*, Badki et al., *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Badki_Bi3D_Stereo_Depth_Estimation_via_Binary_Classifications_CVPR_2020_paper.pdf)] [[Code](https://github.com/NVlabs/Bi3D)] [[Bibtex](./bibliography/Bi3D.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Bi3D:+Stereo+Depth+Estimation+via+Binary+Classifications&btnG=)]

  </details>


  <details open class="nested-details">
    <summary style="font-size: larger; font-weight: bold;">3D Architectures</summary>

   * :triangular_flag_on_post: **GC-Net**: *"End-to-end learning of geometry and context for deep stereo regression"*, Kendall et al., *ICCV, 2017*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Kendall_End-To-End_Learning_of_ICCV_2017_paper.pdf)] [[Bibtex](./bibliography/GC-Net.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=End-to-end+learning+of+geometry+and+context+for+deep+stereo+regression&btnG=)]

   * **ECA**: *"Deep stereo matching with explicit cost aggregation sub-architecture"*, Yu et al., *AAAI, 2018*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/12267/12126)] [[Bibtex](./bibliography/ECA.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Deep+stereo+matching+with+explicit+cost+aggregation+sub-architecture&btnG=)]

   * **PSMNet**: *"Pyramid Stereo Matching Network"*, Chang et al., *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chang_Pyramid_Stereo_Matching_CVPR_2018_paper.pdf)] [[Code](https://github.com/JiaRenChang/PSMNet)] [[Bibtex](./bibliography/PSMNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Pyramid+Stereo+Matching+Network&btnG=)]

   * **PDSNet**: *"Practical deep stereo (pds): Toward applications-friendly deep stereo matching"*, Tulyakov et al., *NeurIPS, 2018*. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/ade55409d1224074754035a5a937d2e0-Paper.pdf)] [[Code](https://github.com/tlkvstepan/PracticalDeepStereo_NIPS2018)] [[Bibtex](./bibliography/PDSNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Practical+deep+stereo+(pds):+Toward+applications-friendly+deep+stereo+matching&btnG=)]

   * **HSMNet**: *"Hierarchical deep stereo matching on high-resolution images"*, Yang et al., *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Hierarchical_Deep_Stereo_Matching_on_High-Resolution_Images_CVPR_2019_paper.pdf)] [[Code](https://github.com/gengshan-y/high-res-stereo)]  [[Bibtex](./bibliography/HSMNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Hierarchical+deep+stereo+matching+on+high-resolution+images&btnG=)]

   * **GWCNet**: *"Group-wise correlation stereo network"*, Guo et al., *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_Group-Wise_Correlation_Stereo_Network_CVPR_2019_paper.pdf)] [[Code](https://github.com/xy-guo/GwcNet)]  [[Bibtex](./bibliography/GWCNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Group-wise+correlation+stereo+network&btnG=)]

   * **EMCUA**: *"Multi-Level Context Ultra-Aggregation for Stereo Matching"*, Nie et al., *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Nie_Multi-Level_Context_Ultra-Aggregation_for_Stereo_Matching_CVPR_2019_paper.pdf)] [[Bibtex](./bibliography/EMCUA.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Multi-Level+Context+Ultra-Aggregation+for+Stereo+Matching&btnG=)]

   * **CSPN**: *"Learning depth with convolutional spatial propagation network"*, Cheng et al., *TPAMI, 2019*. [[Paper](https://arxiv.org/pdf/1810.02695.pdf)] [[Code](https://github.com/XinJCheng/CSPN)] [[Bibtex](./bibliography/CSPN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+depth+with+convolutional+spatial+propagation+network&btnG=)]

   * **GA-Net**: *"Ga-net: Guided aggregation net for end-to-end stereo matching"*, Zhang et al., *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_GA-Net_Guided_Aggregation_Net_for_End-To-End_Stereo_Matching_CVPR_2019_paper.pdf)] [[Code](https://github.com/feihuzhang/GANet)] [[Bibtex](./bibliography/GANet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Ga-net:+Guided+aggregation+net+for+end-to-end+stereo+matching&btnG=)]

   * **Stereodrnet**: *"Stereodrnet: Dilated residual stereonet"*, Chabra et al., *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chabra_StereoDRNet_Dilated_Residual_StereoNet_CVPR_2019_paper.pdf)] [[Bibtex](./bibliography/Stereodrnet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Stereodrnet:+Dilated+residual+stereonet&btnG=)]

   * **CasStereo**: *"Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching"*, Gu et al., *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gu_Cascade_Cost_Volume_for_High-Resolution_Multi-View_Stereo_and_Stereo_Matching_CVPR_2020_paper.pdf)] [[Code](https://github.com/alibaba/cascade-stereo)]  [[Bibtex](./bibliography/CasStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Cascade+Cost+Volume+for+High-Resolution+Multi-View+Stereo+and+Stereo+Matching&btnG=)]

   * **WaveletStereo**: *"WaveletStereo: Learning Wavelet Coefficients of Disparity Map in Stereo Matching"*, Wang et al., *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_WaveletStereo_Learning_Wavelet_Coefficients_of_Disparity_Map_in_Stereo_Matching_CVPR_2020)] [[Bibtex](./bibliography/WaveletStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=WaveletStereo:+Learning+Wavelet+Coefficients+of+Disparity+Map+in+Stereo+Matching&btnG=)]

   * **CFNet**: *"CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching"*, Shen et al., *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Shen_CFNet_Cascade_and_Fused_Cost_Volume_for_Robust_Stereo_Matching_CVPR_2021_paper.pdf)] [[Code](https://github.com/gallenszl/CFNet)] [[Bibtex](./bibliography/CFNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=CFNet:+Cascade+and+Fused+Cost+Volume+for+Robust+Stereo+Matching&btnG=)]

   * **UASNet**: *"UASNet: Uncertainty Adaptive Sampling Network for Deep Stereo Matching"*, Mao et al., *ICCV, 2021* [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Mao_UASNet_Uncertainty_Adaptive_Sampling_Network_for_Deep_Stereo_Matching_ICCV_2021_paper.pdf)] [[Bibtex](./bibliography/UASNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=UASNet:+Uncertainty+Adaptive+Sampling+Network+for+Deep+Stereo+Matching&btnG=)]

   * **PCR**: *"Parallax contextual representations for stereo matching"*, Deng et al., *ICIP, 2021*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9506747)] [[Bibtex](./bibliography/PCR.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=PARALLAX+CONTEXTUAL+REPRESENTATIONS+FOR+STEREO+MATCHING&btnG=)]

   * **PCWNet**: *"PCW-Net: Pyramid Combination and Warping Cost Volume for Stereo Matching"*, Shen et al., *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920280.pdf)] [[Code](https://github.com/gallenszl/PCWNet)] [[Bibtex](./bibliography/PCWNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=PCW-Net:+Pyramid+Combination+and+Warping+Cost+Volume+for+Stereo+Matching&btnG=)]

   * **ICVP**: *"Image-Coupled Volume Propagation for Stereo Matching"*, Kwon et al., *ICIP, 2023*. [[Paper](https://ieeexplore.ieee.org/abstract/document/10222247)] [[Code](https://github.com/ohkwon718/icvp)] [[Bibtex](./bibliography/ICVP.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Image-Coupled+Volume+Propagation+for+Stereo+Matching&btnG=)]

   * **SEDNet**: *"Learning the distribution of errors in stereo matching for joint disparity and uncertainty estimation"*, Chen et al., CVPR, 2023. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Learning_the_Distribution_of_Errors_in_Stereo_Matching_for_Joint_CVPR_2023_paper.pdf)] [[Code](https://github.com/lly00412/SEDNet)] [[Bibtex](./bibliography/SEDNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+the+distribution+of+errors+in+stereo+matching+for+joint+disparity+and+uncertainty+estimation&btnG=)]

   * **Coatrsnet**: *"Coatrsnet: Fully exploiting convolution and attention for stereo matching by region separation"*, Junda et al., IJCV, 2024. [[Paper](https://link.springer.com/content/pdf/10.1007/s11263-023-01872-0.pdf)] [[Bibtex](./bibliography/Coatrsnet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Coatrsnet%3A+Fully+exploiting+convolution+and+attention+for+stereo+matching+by+region+separation&btnG=)]

  </details>

  </details>


  <details open class="nested-details">
    <summary style="font-size: larger; font-weight: bold;">Neural Architecture Search (NAS)</summary>

   * **LEAStereo**: *"Hierarchical Neural Architecture Search for Deep Stereo Matching"*, Cheng et al., *NeurIPS, 2020*. [[Paper](https://arxiv.org/pdf/2010.13501.pdf)] [[Code](https://github.com/XuelianCheng/LEAStereo)] [[Bibtex](./bibliography/LEAStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Hierarchical+Neural+Architecture+Search+for+Deep+Stereo+Matching&btnG=)]

   * **EASNet**: *"EASNet: searching elastic and accurate network architecture for stereo matching"*, Wang et al., *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920434.pdf)] [[Code](https://github.com/HKBU-HPML/EASNet)] [[Bibtex](./bibliography/EASNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=EASNet:+searching+elastic+and+accurate+network+architecture+for+stereo+matching&btnG=)]

   </details>

  <details open class="nested-details">
    <summary style="font-size: larger; font-weight: bold;">Iterative Optimized-based Architectures</summary>
    
   * :triangular_flag_on_post: **RAFT-Stereo**: *"RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching"*, Lipson et al., *3DV, 2021*. [[Paper](https://arxiv.org/abs/2109.07547)] [[Code](https://github.com/princeton-vl/RAFT-Stereo)] [[Bibtex](./bibliography/RAFT-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=RAFT-Stereo:+Multilevel+Recurrent+Field+Transforms+for+Stereo+Matching&btnG=)]

   * **ORStereo**: *"Orstereo: Occlusion-aware recurrent stereo matching for 4k-resolution images"*, Hu et al., *IROS, 2021*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9635869&casa_token=xOCL674auMgAAAAA:cMdpD6Bfa_fmltkjCO2Ig2snwGn12h5vZuYIp0emUAe-CqQy0uO82F7vWOpCdaeg-xmeOi__&tag=1)] [[WebPage](https://theairlab.org/orstereo/)] [[Bibtex](./bibliography/ORStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=ORStereo%3A+Occlusion-Aware+Recurrent+Stereo+Matching+for+4K-Resolution+Images&btnG=)]

   * **SCV-Stereo**: *"SCV-Stereo: Learning Stereo Matching from a Sparse Cost Volume"*, Wang et al., *ICIP, 2021*. [[Paper](https://arxiv.org/abs/2107.08187)] [[Code](https://sites.google.com/view/scv-stereo)] [[Bibtex](./bibliography/SCV-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=SCV-Stereo:+Learning+Stereo+Matching+from+a+Sparse+Cost+Volume&btnG=)] 

   * **CREStereo**: *"Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation"*, Li et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Practical_Stereo_Matching_via_Cascaded_Recurrent_Network_With_Adaptive_Correlation_CVPR_2022_paper.pdf)] [[Code](https://github.com/megvii-research/CREStereo)] [[Bibtex](./bibliography/CREStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Practical+Stereo+Matching+via+Cascaded+Recurrent+Network+with+Adaptive+Correlation&btnG=)]

   * **EAI-Stereo**: *"EAI-Stereo: Error Aware Iterative Network for Stereo Matching"*, Zhao et al., *ACCV, 2022*. [[Paper](https://openaccess.thecvf.com/content/ACCV2022/html/Zhao_EAI-Stereo_Error_Aware_Iterative_Network_for_Stereo_Matching_ACCV_2022_paper.html)] [[Code](https://github.com/smartadpole/EAI-Stereo)] [[Bibtex](./bibliography/EAI-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=EAI-Stereo:+Error+Aware+Iterative+Network+for+Stereo+Matching&btnG=)]

   * **IGEV-Stereo**: *"Iterative Geometry Encoding Volume for Stereo Matching"*, Xu et al., *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Iterative_Geometry_Encoding_Volume_for_Stereo_Matching_CVPR_2023_paper.pdf)] [[Code](https://github.com/gangweiX/IGEV)] [[Bibtex](./bibliography/IGEV-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Iterative+Geometry+Encoding+Volume+for+Stereo+Matching&btnG=)]

   * **DLNR**: *"High-Frequency Stereo Matching Network"*, Zhao et al, *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_High-Frequency_Stereo_Matching_Network_CVPR_2023_paper.pdf)] [[Code](https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network)] [[Bibtex](./bibliography/DLNR.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=High-Frequency+Stereo+Matching+Network&btnG=)]

   * **Dynamic Stereo**: *"DynamicStereo: Consistent Dynamic Depth From Stereo Videos"*, Karaev et al., *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Karaev_DynamicStereo_Consistent_Dynamic_Depth_From_Stereo_Videos_CVPR_2023_paper.pdf)] [[Code](https://dynamic-stereo.github.io/)] [[Bibtex](./bibliography/Dynamic_Replica.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=DynamicStereo:+Consistent+Dynamic+Depth+From+Stereo+Videos&btnG=)]

   * **CREStereo++**: *"Uncertainty Guided Adaptive Warping for Robust and Efficient Stereo Matching"*, Jing et al., *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Jing_Uncertainty_Guided_Adaptive_Warping_for_Robust_and_Efficient_Stereo_Matching_ICCV_2023_paper.pdf)] [[Bibtex](./bibliography/CREStereo++.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Uncertainty+Guided+Adaptive+Warping+for+Robust+and+Efficient+Stereo+Matching&btnG=)]

   * **Selective-Stereo**: *"Selective-Stereo: Adaptive Frequency Information Selection for Stereo Matching"*, Wang et al., *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2403.00486.pdf)] [[Code](https://github.com/Windsrain/Selective-Stereo)] [[Bibtex](./bibliography/Selective-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Selective-Stereo:+Adaptive+Frequency+Information+Selection+for+Stereo+Matching&btnG=)]

   * **Any-Stereo**: *"Any-Stereo: Arbitrary Scale Disparity Estimation for Iterative Stereo Matching"*, Liang et al., *AAAI, 2024*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28119/28242)] [[Code](https://github.com/Zhaohuai-L/Any-Stereo)] [[Bibtex](./bibliography/Any-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Any-Stereo:+Arbitrary+Scale+Disparity+Estimation+for+Iterative+Stereo+Matching&btnG=)]

   * **MC-Stereo**: *"MC-Stereo: Multi-peak Lookup and Cascade Search Range for Stereo Matching"*, Feng et al., *3DV, 2024*. [[Paper](https://arxiv.org/pdf/2311.02340.pdf)] [[Code](https://github.com/MiaoJieF/MC-Stereo)] [[Bibtex](./bibliography/MC-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=MC-Stereo:+Multi-peak+Lookup+and+Cascade+Search+Range+for+Stereo+Matching&btnG=)]

   * **ICGNet**: *"Learning Intra-view and Cross-view Geometric Knowledge for Stereo Matching"*, Gong et al., *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2402.19270.pdf)] [[Code](https://github.com/DFSDDDDD1199/ICGNet)] [[Bibtex](./bibliography/ICGNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+Intra-view+and+Cross-view+Geometric+Knowledge+for+Stereo+Matching&btnG=)]

   * **MoCha-Stereo**: *"MoCha-Stereo: Motif Channel Attention Network for Stereo Matching"*, Chen et al., *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2404.06842.pdf)] [[Code](https://github.com/ZYangChen/MoCha-Stereo)] [[Bibtex](./bibliography/MoCha-Stereo.txt)] [[Google Scholar](https://scholar.google.de/scholar?q=MoCha-Stereo:%20Motif%20Channel%20Attention%20Network%20for%20Stereo%20Matching)]


   * **LoS**: *"LoS: Local Structure-guided Stereo Matching"*, Li et al., *CVPR, 2024*. [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_LoS_Local_Structure-Guided_Stereo_Matching_CVPR_2024_paper.pdf)] [[Code](https://github.com/SYSU-SAIL/LoS)] [[Bibtex](./bibliography/LoS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=LoS%3A+Local+Structure-guided+Stereo+Matching&btnG=)]

   * **XR-Stereo**: *"Stereo Matching in Time: 100+ FPS Video Stereo Matching for Extended Reality"*, Cheng et al., *WACV, 2024*. [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Cheng_Stereo_Matching_in_Time_100_FPS_Video_Stereo_Matching_for_WACV_2024_paper.pdf)] [[Code](https://github.com/za-cheng/XR-Stereo)] [[Bibtex](./bibliography/XR-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Stereo+Matching+in+Time:+100++FPS+Video+Stereo+Matching+for+Extended+Reality&btnG=)]

   * **Temporally-Consistent Stereo**: *"Temporally Consistent Stereo Matching"*, Zeng et al., *ECCV, 2024*. [[Paper](https://arxiv.org/pdf/2407.11950)] [[Code](https://github.com/jiaxiZeng/Temporally-Consistent-Stereo-Matching)] [[Bibtex](./bibliography/Temporally-Consistent-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Temporally+Consistent+Stereo+Matching&btnG=)]

   * **BiDA-Stereo**: *"Match-Stereo-Videos: Bidirectional Alignment for Consistent Dynamic Stereo Matching"*, Jing et al., *ECCV, 2024*. [[Paper](https://arxiv.org/abs/2403.10755)] [[Code](https://github.com/TomTomTommi/bidastereo)] [[Bibtex](./bibliography/Match-Stereo-Videos.txt)] [[WebPage](https://tomtomtommi.github.io/BiDAStereo/)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Match-Stereo-Videos%3A+Bidirectional+Alignment+for+Consistent+Dynamic+Stereo+Matching&btnG=)]

   * **QPDNet**: *"Disparity Estimation Using a Quad-Pixel Sensor"*, Wu et al., *BMVC, 2024*. [[Paper](https://arxiv.org/abs/2409.00665)] [[WebPage](http://www.ok.sc.e.titech.ac.jp/res/QP/index.html)] [[Dataset](https://www.dropbox.com/scl/fi/0sijedbg2kdqetmw16s9t/QP-Data-Zhuofeng-Wu.zip?rlkey=m33xsb70b4ivk0riyvg4fgk2u&st=k2fn4zkq&dl=0)] [[Bibtex](./bibliography/QPDNet.txt)]  [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Disparity+Estimation+Using+a+Quad-Pixel+Sensor&btnG=)]

   * **IGEV++**: *"IGEV++: Iterative Multi-range Geometry Encoding Volumes for Stereo Matching"*, Xu et al., *arXiv, 2024*. [[Paper](https://arxiv.org/pdf/2409.00638)] [[Code](https://github.com/gangweiX/IGEV-plusplus)] [[Bibtex](./bibliography/IGEV++.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=IGEV%2B%2B%3A+Iterative+Multi-range+Geometry+Encoding+Volumes+for+Stereo+Matching&btnG=)]

   * **Stereo Anything**: *"Stereo Anything: Unifying Stereo Matching with Large-Scale Mixed Data"*, Guo et al., *arXiv, 2024*. [[Paper](https://arxiv.org/pdf/2411.14053)] [[Code](https://github.com/XiandaGuo/OpenStereo)] [[Bibtex](./bibliography/StereoAnything.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Stereo+Anything%3A+Unifying+Stereo+Matching+with+Large-Scale+Mixed+Data&btnG=)]

   * **MonSter**: *"MonSter: Marry Monodepth to Stereo Unleashes Power"*, Cheng et al., *CVPR, 2025*. [[Paper](https://arxiv.org/pdf/2501.08643)] [[Code](https://github.com/Junda24/MonSter)] [[Bibtex](./bibliography/MonSter.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=MonSter%3A+Marry+Monodepth+to+Stereo+Unleashes+Power&btnG=)]

   * **Stereo Anywhere**: *"Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching Even Where Either Stereo or Mono Fail"*, Bartolomei et al., *CVPR, 2025*. [[Paper](http://arxiv.org/abs/2412.04472)] [[Code](https://github.com/bartn8/stereoanywhere/)] [[WebPage](https://stereoanywhere.github.io/)] [[Bibtex](./bibliography/StereoAnywhere.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Stereo+Anywhere%3A+Robust+Zero-Shot+Deep+Stereo+Matching+Even+Where+Either+Stereo+or+Mono+Fail&btnG=)]

   * **FoundationStereo**: *"FoundationStereo: Zero-Shot Stereo Matching"*, Wen et al., *CVPR, 2025*. [[Paper](https://arxiv.org/pdf/2501.09898)] [[Code](https://github.com/NVlabs/FoundationStereo)] [[WebPage](https://nvlabs.github.io/FoundationStereo/)] [[Bibtex](./bibliography/FoundationStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=FoundationStereo%3A+Zero-Shot+Stereo+Matching&btnG=#d=gs_cit&t=1737624791731&u=%2Fscholar%3Fq%3Dinfo%3AmPuArdM7kccJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Dit)]

   * **DEFOM-Stereo**: *"DEFOM-Stereo: Depth Foundation Model Based Stereo Matching"*, Jiang et al., *arXiv, 2025*. [[Paper](https://arxiv.org/pdf/2501.09466)] [[Code](https://github.com/Insta360-Research-Team/DEFOM-Stereo)] [[Bibtex](./bibliography/DEFOM-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=DEFOM-Stereo%3A+Depth+Foundation+Model+Based+Stereo+Matching&btnG=)]

   * **AIO-Stereo**: *"All-in-One: Transferring Vision Foundation Models into Stereo Matching"*, Zhou et al., *AAAI, 2025*. [[Paper](https://arxiv.org/pdf/2412.09912)] [[Bibtex](./bibliography/AIO-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=All-in-One%3A+Transferring+Vision+Foundation+Models+into+Stereo+Matching&btnG=)]


  </details>


  <details open class="nested-details">
    <summary style="font-size: larger; font-weight: bold;">Transformer-based Architectures</summary>
    
   * **STTR**: *"Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective With Transformers"*, Li et al., *ICCV, 2021*  [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Revisiting_Stereo_Depth_Estimation_From_a_Sequence-to-Sequence_Perspective_With_Transformers_ICCV_2021_paper.pdf)] [[Code](https://github.com/mli0603/stereo-transformer)] [[Bibtex](./bibliography/STTR.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Revisiting+Stereo+Depth+Estimation+From+a+Sequence-to-Sequence+Perspective+With+Transformers&btnG=)]

   * **CEST**: *"Context-enhanced stereo transformer"*, Guo et al., *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920263.pdf)] [[Code](https://github.com/guoweiyu/Context-Enhanced-Stereo-Transformer)] [[Bibtex](./bibliography/CEST.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Context-enhanced+stereo+transformer&btnG=)]

   * **Chitransformer**: *"Chitransformer: Towards Reliable Stereo From Cues"*, Su et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Su_Chitransformer_Towards_Reliable_Stereo_From_Cues_CVPR_2022_paper.pdf)] [[Code](https://github.com/ISL-CV/ChiTransformer)] [[Bibtex](./bibliography/Chitransformer.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Chitransformer:+Towards+Reliable+Stereo+From+Cues&btnG=)]

   * **Dynamic Stereo**: *"DynamicStereo: Consistent Dynamic Depth From Stereo Videos"*, Karaev et al., *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Karaev_DynamicStereo_Consistent_Dynamic_Depth_From_Stereo_Videos_CVPR_2023_paper.pdf)] [[Code](https://dynamic-stereo.github.io/)] [[Bibtex](./bibliography/Dynamic_Replica.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=DynamicStereo:+Consistent+Dynamic+Depth+From+Stereo+Videos&btnG=)]
   
   * **GMStereo**: *"Unifying Flow, Stereo and Depth Estimation"*, Xu et al., *TPAMI, 2023*. [[Paper](https://arxiv.org/pdf/2211.05783.pdf)] [[Code](https://haofeixu.github.io/unimatch/)] [[Bibtex](./bibliography/GMStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Unifying+Flow,+Stereo+and+Depth+Estimation&btnG=)]

   * **CroCo v2**: *"CroCo v2: Improved Cross-View Completion Pre-training for Stereo Matching and Optical Flow"*, Weinzaepfel et al., *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Weinzaepfel_CroCo_v2_Improved_Cross-view_Completion_Pre-training_for_Stereo_Matching_and_ICCV_2023_paper.pdf)] [[Code](https://github.com/naver/croco)] [[Bibtex](./bibliography/CroCo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=CroCo+v2:+Improved+Cross-View+Completion+Pre-training+for+Stereo+Matching+and+Optical+Flow&btnG=)]

   * **ELFNet**: *"Elfnet: Evidential local-global fusion for stereo matching"*, Lou et al., *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lou_ELFNet_Evidential_Local-global_Fusion_for_Stereo_Matching_ICCV_2023_paper.pdf)] [[Code](https://openaccess.thecvf.com/content/ICCV2023/papers/Lou_ELFNet_Evidential_Local-global_Fusion_for_Stereo_Matching_ICCV_2023_paper.pdf)] [[Bibtex](./bibliography/ELFNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Elfnet:+Evidential+local-global+fusion+for+stereo+matching&btnG=)]

   * **GOAT**: *"Global Occlusion-Aware Transformer for Robust Stereo Matching"*, Liu et al., *WACV, 2024*. [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Liu_Global_Occlusion-Aware_Transformer_for_Robust_Stereo_Matching_WACV_2024_paper.pdf)] [[Code](https://github.com/Magicboomliu/GOAT)] [[Bibtex](./bibliography/GOAT.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Global+Occlusion-Aware+Transformer+for+Robust+Stereo+Matching&btnG=)]

   * **FormerStereo**: *"Learning Representations from Foundation Models for Domain Generalized Stereo Matching"*, Zhang et al., *ECCV, 2024*. [[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05841.pdf)] [[Bibtex](./bibliography/FormerStereo.txt)]   [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Learning+Representations+from+Foundation+Models+for+Domain+Generalized+Stereo+Matching&btnG=)]

   * **ViTAStereo**: *"Playing to Vision Foundation Model’s Strengths in Stereo Matching"*, Zhang et al., *T-IV, 2024*. [[Paper](https://arxiv.org/pdf/2404.06261)] [[Code](https://github.com/CwLiuzzZ/ViTAS)] [[Bibtex](./bibliography/ViTAStereo.txt)]   [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Playing+to+Vision+Foundation+Model%E2%80%99s+Strengths+in+Stereo+Matching&btnG=)]

  </details>

  <details open class="nested-details">
    <summary style="font-size: larger; font-weight: bold;">Markov Random Field-based Architectures</summary>

   * **LBPS**: *"Belief Propagation Reloaded: Learning BP-Layers for Labeling Problems"*, Knobelreiter et al., *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Knobelreiter_Belief_Propagation_Reloaded_Learning_BP-Layers_for_Labeling_Problems_CVPR_2020_paper.pdf)] [[Code](https://github.com/VLOGroup/bp-layers)] [[Bibtex](./bibliography/LBPS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Belief+propagation+reloaded%3A+Learning+bp-layers+for+labeling+problem&btnG=)]

   * **NMRF**: *"Neural Markov Random Field for Stereo Matching"*, Guan et al., *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2403.11193.pdf)] [[Code](https://github.com/aeolusguan/NMRF)] [[Bibtex](./bibliography/NMRF.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Neural+Markov+Random+Field+for+Stereo+Matching&btnG=)]

  </details>

</ul>
</details>

<details open id="efficient-oriented">
<summary style="font-size: larger; font-weight: bold;">Efficient-Oriented Deep Stereo Architectures</summary><ul>

<details open>
<summary style="font-size: larger; font-weight: bold;">Compact Cost Volume Representation</summary>

* **Stereonet**: *"Stereonet: Guided hierarchical refinement for real-time edge-aware depth prediction"*, Khamis et al., *ECCV, 2018*. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sameh_Khamis_StereoNet_Guided_Hierarchical_ECCV_2018_paper.pdf)] [[Code](https://github.com/neka-nat/StereoNet)] [[Bibtex](./bibliography/Stereonet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Stereonet:+Guided+hierarchical+refinement+for+real-time+edge-aware+depth+prediction&btnG=)] 

* **Fast DS-CS**: *"Fast Deep Stereo with 2D Convolutional Processing of Cost Signatures"*, Yee et al., *WACV, 2020* [[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Yee_Fast_Deep_Stereo_with_2D_Convolutional_Processing_of_Cost_Signatures_WACV_2020_paper.pdf)]  [[Code](https://github.com/ayanc/fdscs)]  [[Bibtex](./bibliography/FDCSC.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Fast+Deep+Stereo+with+2D+Convolutional+Processing+of+Cost+Signatures&btnG=)]

* **DecNet**: *"A Decomposition Model for Stereo Matching"*, Yao et al., *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yao_A_Decomposition_Model_for_Stereo_Matching_CVPR_2021_paper.pdf)] [[Code](https://github.com/YaoChengTang/DecNet)] [[Bibtex](./bibliography/DecNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=A+Decomposition+Model+for+Stereo+Matching&btnG=)]


* **BTC**: *"Soft Cross Entropy Loss and Bottleneck Tri-Cost Volume For Efficient Stereo Depth Prediction"*, Nuanes et al., *CVPRW, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021W/WAD/papers/Nuanes_Soft_Cross_Entropy_Loss_and_Bottleneck_Tri-Cost_Volume_for_Efficient_CVPRW_2021_paper.pdf)]  [[Bibtex](./bibliography/BTC.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Soft+Cross+Entropy+Loss+and+Bottleneck+Tri-Cost+Volume+for+Efficient+Stereo+Depth+Prediction+&btnG=)]


* **ACVNet**: *"Attention Concatenation Volume for Accurate and Efficient Stereo Matching"*, Xu et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Attention_Concatenation_Volume_for_Accurate_and_Efficient_Stereo_Matching_CVPR_2022_paper.pdf)] [[Code](https://github.com/gangweiX/ACVNet)] [[Bibtex](./bibliography/ACVNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Attention+Concatenation+Volume+for+Accurate+and+Efficient+Stereo+Matching&btnG=)]

* **PCVNet**: *"Parameterized Cost Volume for Stereo Matching"*, Zeng et al., *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zeng_Parameterized_Cost_Volume_for_Stereo_Matching_ICCV_2023_paper.pdf)] [[Code](https://github.com/jiaxiZeng/Parameterized-Cost-Volume-for-Stereo-Matching)] [[Bibtex](./bibliography/PCVNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Parameterized+Cost+Volume+for+Stereo+Matching&btnG=)]

* **IINet**: *"IINet: Implicit Intra-inter Information Fusion for Real-Time Stereo Matching"*, Li et al., *AAAI, 2024*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28107/28218)] [[Code](https://github.com/blindwatch/IINet)] [[Bibtex](./bibliography/IINet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=IINet:+Implicit+Intra-inter+Information+Fusion+for+Real-Time+Stereo+Matching&btnG=)]


</details>


<details open>
<summary style="font-size: larger; font-weight: bold;">Efficient Cost Volume Processing</summary>

* **Deeppruner**: *"Deeppruner: Learning efficient stereo matching via differentiable patchmatch"*, Duggal et al., *ICCV, 2019*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Duggal_DeepPruner_Learning_Efficient_Stereo_Matching_via_Differentiable_PatchMatch_ICCV_2019_paper.pdf)] [[Code](https://github.com/uber-research/DeepPruner)] [[Bibtex](./bibliography/Deeppruner.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Deeppruner:+Learning+efficient+stereo+matching+via+differentiable+patchmatch&btnG=)] 

* **CasStereo**: *"Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching"*, Gu et al., *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gu_Cascade_Cost_Volume_for_High-Resolution_Multi-View_Stereo_and_Stereo_Matching_CVPR_2020_paper.pdf)] [[Code](https://github.com/alibaba/cascade-stereo)]  [[Bibtex](./bibliography/CasStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Cascade+Cost+Volume+for+High-Resolution+Multi-View+Stereo+and+Stereo+Matching&btnG=)]

* **MABNet**: *"MABNet: a lightweight stereo network based on multibranch adjustable bottleneck module"*, Xing et al.,  *ECCV, 2020*. [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730341.pdf)] [[Code](https://github.com/JumpXing/MABNet)] [[Bibtex](./bibliography/MABNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=MABNet:+a+lightweight+stereo+network+based+on+multibranch+adjustable+bottleneck+module&btnG=)] 

* **BGNet**: *"Bilateral Grid Learning for Stereo Matching Networks"*, Xu et al., *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Bilateral_Grid_Learning_for_Stereo_Matching_Networks_CVPR_2021_paper.pdf)] [[Code](https://github.com/3DCVdeveloper/BGNet)] [[Bibtex](./bibliography/BGNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Bilateral+Grid+Learning+for+Stereo+Matching+Networks&btnG=)]

* **Separable-Stereo**: *"Separable Convolutions for Optimizing 3D Stereo Networks"*, Rahim et al., *ICIP, 2021*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9506330)] [[Code](https://github.com/cogsys-tuebingen/separable-3D-convs-for-stereo-matching)] [[Bibtex](./bibliography/Separable-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Separable+Convolutions+for+Optimizing+3D+Stereo+Networks&btnG=)]

* **TemporalStereo**: *"TemporalStereo: Efficient Spatial-Temporal Stereo Matching Network"*, Zhang et al., *IROS, 2023*. [[Paper](https://arxiv.org/pdf/2211.13755.pdf)] [[Code](https://github.com/youmi-zym/TemporalStereo)]  [[Bibtex](./bibliography/TemporalStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=TemporalStereo:+Efficient+Spatial-Temporal+Stereo+Matching+Network&btnG=)]


</details>


<details open>
<summary style="font-size: larger; font-weight: bold;">Efficient Inference Schemes</summary>

* **Anytime**: *"Anytime stereo image depth estimation on mobile devices"*, Wang et al., *ICRA, 2019*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8794003)] [[Code](https://github.com/mileyan/AnyNet)] [[Bibtex](./bibliography/Anytime.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Anytime+stereo+image+depth+estimation+on+mobile+devices&btnG=)] 

* **StereoVAE**: *"StereoVAE: A lightweight stereo-matching system using embedded GPUs"*, Chang et al., *ICRA, 2023*. [[Paper](https://ieeexplore.ieee.org/document/10160441)] [[Bibtex](./bibliography/StereoVAE.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=StereoVAE:+A+lightweight+stereo-matching+system+using+embedded+GPUs&btnG=)]

* **ADStereo**: *"ADStereo: Efficient Stereo Matching With Adaptive Downsampling and Disparity Alignment"*, Wang et al., *TIP, 2023*. [[Paper](https://ieeexplore.ieee.org/abstract/document/10890914)] [[Code](https://github.com/cocowy1/ADStereo)] [[Bibtex](./bibliography/ADStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=ADStereo%3A+Efficient+Stereo+Matching+With+Adaptive+Downsampling+and+Disparity+Alignment&btnG=)]


</details>


<details open>
<summary style="font-size: larger; font-weight: bold;">Lightweight Network Architecture Design</summary>

* **NVStereoNet**: *"On the importance of stereo for accurate depth estimation: An efficient semi-supervised deep neural network approach"*, Smolyanskiy et al., *CVPRW, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w14/Smolyanskiy_On_the_Importance_CVPR_2018_paper.pdf)] [[Code](https://github.com/NVIDIA-AI-IOT/redtail)] [[Bibtex](./bibliography/NVStereoNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=On+the+Importance+of+Stereo+for+Accurate+Depth+Estimation%3A+An+Efficient+Semi-Supervised+Deep+Neural+Network+Approach&btnG=)]

* **MadNet**: *"Real-Time Self-Adaptive Deep Stereo"*, Tonioni et al., *CVPR, 2019*. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tonioni_Real-Time_Self-Adaptive_Deep_Stereo_CVPR_2019_paper.pdf)] [[Code](https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo?tab=readme-ov-file)] [[Bibtex](./bibliography/MadNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Real-Time+Self-Adaptive+Deep+Stereo&btnG=)]

* **Fadnet**: *"Fadnet: A Fast and Accurate Network for Disparity Estimation"*, Wang et al., *ICRA, 2020*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9197031)] [[Code](https://github.com/HKBU-HPML/FADNet)] [[Bibtex](./bibliography/Fadnet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Fadnet:+A+Fast+and+Accurate+Network+for+Disparity+Estimation&btnG=)]

* **AAFS**: *"Attention-Aware Feature Aggregation for Real-time Stereo Matching on Edge Devices"*, Chang et al., *ACCV, 2020* [[Code](https://github.com/JiaRenChang/RealtimeStereo)] [[Paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Chang_Attention-Aware_Feature_Aggregation_for_Real-time_Stereo_Matching_on_Edge_Devices_ACCV_2020_paper.pdf)] [[Bibtex](./bibliography/AAFS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Attention-Aware+Feature+Aggregation+for+Real-time+Stereo+Matching+on+Edge+Devices&btnG=)]

* **HITNet**: *"HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching"*, Tankovich et al., *CVPR, 2021*. [[Paper](https://arxiv.org/abs/2007.12140)] [[Code](https://github.com/google-research/google-research/tree/master/hitnet)] [[Bibtex](./bibliography/HITNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=HITNet:+Hierarchical+Iterative+Tile+Refinement+Network+for+Real-time+Stereo+Matching&btnG=)]

* **CoEX**: *"Correlate-and-Excite: Real-Time Stereo Matching via Guided Cost Volume Excitation"*, Bangunharcana et al., *IROS, 2021*. [[Paper](https://antabangun.github.io/projects/CoEx/)] [[Code](https://github.com/antabangun/coex)] [[Bibtex](./bibliography/CoEX.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Correlate-and-Excite:+Real-Time+Stereo+Matching+via+Guided+Cost+Volume+Excitation&btnG=)]

* **RLStereo**: *"RLStereo: Real-time stereo matching based on reinforcement learning"*, Yang et al., *TIP, 2021*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9614986)] [[Bibtex](./bibliography/RLStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=RLStereo:+Real-time+stereo+matching+based+on+reinforcement+learning&btnG=)]

* **MobileStereoNet**: *"MobileStereoNet: Towards Lightweight Deep Networks for Stereo Matching"*, Shamsafar et al., *WACV, 2022*. [[Paper](https://openaccess.thecvf.com/content/WACV2022/papers/Shamsafar_MobileStereoNet_Towards_Lightweight_Deep_Networks_for_Stereo_Matching_WACV_2022_paper.pdf)] [[Code](https://github.com/cogsys-tuebingen/mobilestereonet)] [[Bibtex](./bibliography/MobileStereoNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=MobileStereoNet:+Towards+Lightweight+Deep+Networks+for+Stereo+Matching&btnG=)]

* **PBCStereo**: *"PBCStereo: A Compressed Stereo Network with Pure Binary Convolutional Operations"*, Cai et al., *ACCV, 2022*. [[Paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Cai_PBCStereo_A_Compressed_Stereo_Network_with_Pure_Binary_Convolutional_Operations_ACCV_2022_paper.pdf)] [[Bibtex](./bibliography/PBCStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=PBCStereo:+A+Compressed+Stereo+Network+with+Pure+Binary+Convolutional+Operations&btnG=)]

* **MadNet2**: *"Federated Online Adaptation for Deep Stereo"*, Poggi et al., *CVPR, 2024*.  [[Paper](https://fedstereo.github.io/)] [[Code](https://github.com/mattpoggi/fedstereo)] [[Bibtex](./bibliography/FedStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Federated+Online+Adaptation+for+Deep+Stereo&btnG=)]

* **Distill-And-Prune**: *"Distill-then-prune: An Efficient Compression Framework for Real-time Stereo Matching Network on Edge Devices"*, Pan et al., *ICRA, 2024*. [[Paper](https://arxiv.org/pdf/2405.11809)] [[Bibtex](./bibliography/Distill-And-Prune.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Distill-then-prune%3A+An+Efficient+Compression+Framework+for+Real-time+Stereo+Matching+Network+on+Edge+Devices&btnG=)]

* **LightStereo**: *"LightStereo: Channel Boost Is All You Need for Efficient 2D Cost Aggregation"*, Guo et al., *ICRA, 2025*. [[Paper](https://arxiv.org/pdf/2406.19833)] [[Code](https://github.com/XiandaGuo/OpenStereo)] [[Bibtex](./bibliography/LightStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=LightStereo%3A+Channel+Boost+Is+All+You+Need+for+Efficient+2D+Cost+Aggregation&btnG=#d=gs_cit&t=1740667250780&u=%2Fscholar%3Fq%3Dinfo%3AMuFhyyW9taIJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Dit)]

</details>












</ul>
</details>


<details open id="multi-task">
<summary style="font-size: larger; font-weight: bold;">Multi-Task Deep Stereo Architectures</summary><ul>

<details open>
<summary style="font-size: larger; font-weight: bold;">Normal-Assisted Stereo Matching</summary>

* **NA-Stereo**: *"Normal Assisted Stereo Depth Estimation"*, Kusupati et al., *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kusupati_Normal_Assisted_Stereo_Depth_Estimation_CVPR_2020_paper.pdf)] [[Code](https://github.com/udaykusupati/Normal-Assisted-Stereo)] [[Bibtex](./bibliography/NA-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Normal+Assisted+Stereo+Depth+Estimation&btnG=)]

* **HITNet**: *"HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching"*, Tankovich et al., *CVPR, 2021*. [[Paper](https://arxiv.org/abs/2007.12140)] [[Code](https://github.com/google-research/google-research/tree/master/hitnet)] [[Bibtex](./bibliography/HITNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=HITNet:+Hierarchical+Iterative+Tile+Refinement+Network+for+Real-time+Stereo+Matching&btnG=)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Joint Stereo Matching and Optical Flow</summary>

* **Multi-Task Learning Using Uncertainty**: *"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"*, Kendall et al., CVPR, 2018. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)]  [[Bibtex](./bibliography/Multi-Task-Learning.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=2005&sciodt=0%2C5&cites=502631552324336717&scipsc=&q=Multi-Task+Learning+Using+Uncertainty+to+Weigh+Losses+for+Scene+Geometry+and+Semantics&btnG=)]


* **BridgeDepthFlow**: *"Bridging Stereo Matching and Optical Flow via Spatiotemporal Correspondence"*, Lai et al., *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lai_Bridging_Stereo_Matching_and_Optical_Flow_via_Spatiotemporal_Correspondence_CVPR_2019_paper.pdf)] [[Code](https://github.com/lelimite4444/BridgeDepthFlow)] [[Bibtex](./bibliography/BridgeDepthFlow.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Bridging+Stereo+Matching+and+Optical+Flow+via+Spatiotemporal+Correspondence&btnG=)]

* **UnOS**: *"UnOS: Unified Unsupervised Optical-Flow and Stereo-Depth Estimation by Watching Videos"*, Wang et al., *CVPR, 2019*. [[Paper](https://ieeexplore.ieee.org/iel7/34/9185119/08769907.pdf)] [[Code](https://github.com/baidu-research/UnDepthflow)]  [[Bibtex](./bibliography/UnOS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=UnOS:+Unified+Unsupervised+Optical-Flow+and+Stereo-Depth+Estimation+by+Watching+Videos&btnG=)]

* **Feature-Level Collaboration**: *"Feature-Level Collaboration: Joint Unsupervised Learning of Optical Flow, Stereo Depth and Camera Motion"*, Chi et al., *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chi_Feature-Level_Collaboration_Joint_Unsupervised_Learning_of_Optical_Flow_Stereo_Depth_CVPR_2021_paper.pdf)] [[Bibtex](./bibliography/Feature-Level_Collaboration.txt)]

* **StereoFlowGAN**: *"StereoFlowGAN: Co-training for Stereo and Flow with Unsupervised Domain Adaptation"*, Xiong et al., *BMVC, 2023*. [[Paper](https://papers.bmvc2023.org/0240.pdf)] [[Bibtex](./bibliography/StereoFlowGAN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=StereoFlowGAN:+Co-training+for+Stereo+and+Flow+with+Unsupervised+Domain+Adaptation&btnG=)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Joint Stereo Matching and Semantic Segmentation</summary>

* **Segstereo**: *"Segstereo: Exploiting semantic information for disparity estimation"*, Yang et al., *ECCV, 2018*. [[Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Guorun_Yang_SegStereo_Exploiting_Semantic_ECCV_2018_paper.pdf)] [[Code](https://github.com/yangguorun/SegStereo)]  [[Bibtex](./bibliography/Segstereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Segstereo:+Exploiting+semantic+information+for+disparity+estimation&btnG=)]

* **Multi-Task Learning Using Uncertainty**: *"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"*, Kendall et al., CVPR, 2018. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)]  [[Bibtex](./bibliography/Multi-Task-Learning.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=2005&sciodt=0%2C5&cites=502631552324336717&scipsc=&q=Multi-Task+Learning+Using+Uncertainty+to+Weigh+Losses+for+Scene+Geometry+and+Semantics&btnG=)]


* **DSNet**: *"DSNet: Joint learning for scene segmentation and disparity estimation"*, Zhan et al., *ICRA, 2019*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8793573)] [[Bibtex](./bibliography/DSNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=DSNet:+Joint+learning+for+scene+segmentation+and+disparity+estimation&btnG=)]

* **Dispsegnet**: *"Dispsegnet: Leveraging semantics for end-to-end learning of disparity estimation from stereo imagery"*, Zhang et al., *RAL, 2019*. [[Paper](https://ieeexplore.ieee.org/iel7/7083369/7339444/08624344.pdf?casa_token=6L9mfcue0jkAAAAA:iMjaxtOTrLvn-9kP5g-NbVDoSAKS5M9LNoIuT3wHecgLmtjrvyhAJ0IXNU4JZjg-4XaOiAsa)] [[Bibtex](./bibliography/Dispsegnet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Dispsegnet:+Leveraging+semantics+for+end-to-end+learning+of+disparity+estimation+from+stereo+imagery&btnG=)]


* **SSPCV-Net**: *"Semantic stereo matching with pyramid cost volumes"*, Wu et al., *ICCV, 2019*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Semantic_Stereo_Matching_With_Pyramid_Cost_Volumes_ICCV_2019_paper.pdf)] [[Bibtex](./bibliography/SSPCV-Net.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Semantic+stereo+matching+with+pyramid+cost+volumes&btnG=)]

* **RSS-Net**: *"Real-time semantic stereo matching"*, Dovesi et al., *ICRA, 2020*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9196784/)] [[Bibtex](./bibliography/RSS-Net.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Real-time+semantic+stereo+matching&btnG=)]

* **SGNet**: *"SGNet: Semantics Guided Deep Stereo Matching"*, Chen et al., *ACCV, 2020*. [[Paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Chen_SGNet_Semantics_Guided_Deep_Stereo_Matching_ACCV_2020_paper.pdf)] [[Bibtex](./bibliography/SGNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=SGNet:+Semantics+Guided+Deep+Stereo+Matching&btnG=)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Joint Stereo Matching and Uncertainty</summary>

   * **RCN**: *"Improved stereo matching with constant highway networks and reflective confidence learning"*, Shaked et al., CVPR, 2017. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Shaked_Improved_Stereo_Matching_CVPR_2017_paper.pdf)] [[Code](https://github.com/amitshaked/resmatch)] [[Bibtex](./bibliography/RCN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Improved+stereo+matching+with+constant+highway+networks+and+reflective+confidence+learning&btnG=)]

   * **UCN**: *"Unified confidence estimation networks for robust stereo matching"*, Kim et al., TIP, 2018. [[Paper](https://ieeexplore.ieee.org/iel7/83/4358840/08510870.pdf)] [[Bibtex](./bibliography/UCN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Unified+confidence+estimation+networks+for+robust+stereo+matching&btnG=)]

   * **ACN**: *"Adversarial confidence estimation networks for robust stereo matching"*, Kim et al.,  T-ITS, 2020. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_LAF-Net_Locally_Adaptive_Fusion_Networks_for_Stereo_Confidence_Estimation_CVPR_2019_paper.pdf)] [[Bibtex](./bibliography/ACN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Adversarial+confidence+estimation+networks+for+robust+stereo+matching&btnG=)]

   * **AcfNet**: *"Adaptive Unimodal Cost Volume Filtering for Deep Stereo Matching"*, Zhang et al., *AAAI, 2020*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6991/6845)] [[Code](https://github.com/youmi-zym/AcfNet)] [[Bibtex](./bibliography/WaveletStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Adaptive+Unimodal+Cost+Volume+Filtering+for+Deep+Stereo+Matching&btnG=)]


   * **Weak Adversarial Learning**: *"Leveraging a weakly adversarial paradigm for joint learning of disparity and confidence estimation"*, Poggi et al., *ICPR, 2021*. [[Paper](https://ieeexplore.ieee.org/iel7/9411940/9411911/09412594.pdf?casa_token=aC1mxhZB1hYAAAAA:ynHT4tbmo7UZSf0kCNTUsbDTsB5BhI-bvKrfWu6PuhkFY27FTPVYeHS7y6qJkeJ9H6AgaatE)] [[Bibtex](./bibliography/WAL.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Leveraging+a+weakly+adversarial+paradigm+for+joint+learning+of+disparity+and+confidence+estimation&btnG=)]

   * **Bayesian**: *"Joint estimation of depth and its uncertainty from stereo images using bayesian deep learning"*, Mehltretter Max, *ISPRS, 2022*. [[Paper](https://isprs-annals.copernicus.org/articles/V-2-2022/69/2022/isprs-annals-V-2-2022-69-2022.pdf)] [[Bibtex](./bibliography/Bayesian.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Joint+estimation+of+depth+and+its+uncertainty+from+stereo+images+using+bayesian+deep+learning&btnG=)]

  * **SEDNet**: *"Learning the distribution of errors in stereo matching for joint disparity and uncertainty estimation"*, Chen et al., CVPR, 2023. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Learning_the_Distribution_of_Errors_in_Stereo_Matching_for_Joint_CVPR_2023_paper.pdf)] [[Code](https://github.com/lly00412/SEDNet)] [[Bibtex](./bibliography/SEDNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+the+distribution+of+errors+in+stereo+matching+for+joint+disparity+and+uncertainty+estimation&btnG=)]

   * **LoS**: *"LoS: Local Structure-guided Stereo Matching"*, Li et al., *CVPR, 2024*. [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_LoS_Local_Structure-Guided_Stereo_Matching_CVPR_2024_paper.pdf)] [[Code](https://github.com/SYSU-SAIL/LoS)] [[Bibtex](./bibliography/LoS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=LoS%3A+Local+Structure-guided+Stereo+Matching&btnG=)]
   
</details>


<details open>
<summary style="font-size: larger; font-weight: bold;"> Scene Flow </summary>

* :triangular_flag_on_post: **FlowNet3.0**: *"Occlusions, motion and depth boundaries with a generic network for disparity, optical flow or scene flow estimation"*, Ilg et al., *ECCV, 2018*. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Eddy_Ilg_Occlusions_Motion_and_ECCV_2018_paper.pdf)] [[Code](https://github.com/lmb-freiburg/netdef_models)] [[Bibtex](./bibliography/FlowNet3.0.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Occlusions,+motion+and+depth+boundaries+with+a+generic+network+for+disparity,+optical+flow+or+scene+flow+estimation&btnG=)]

* **DRISF**: *"Deep Rigid Instance Scene Flow"*, Ma et al., *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ma_Deep_Rigid_Instance_Scene_Flow_CVPR_2019_paper.pdf)] [[Bibtex](./bibliography/DRISF.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Deep+Rigid+Instance+Scene+Flow&btnG=)]

* **DeblurringSF**: *"Joint stereo video deblurring, scene flow estimation and moving object segmentation"*, Pan et al., *TIP, 2019*. [[Paper](https://ieeexplore.ieee.org/iel7/83/4358840/08866754.pdf)] [[Bibtex](./bibliography/DeblurringSF.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Joint+stereo+video+deblurring,+scene+flow+estimation+and+moving+object+segmentation&btnG=)]

* **IOSF**: *"Learning Independent Object Motion From Unlabelled Stereoscopic Videos"*, Cao et al., *TPAMI, 2019*. [[Paper](IOSF )] [[Bibtex](./bibliography/IOSF.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+Independent+Object+Motion+From+Unlabelled+Stereoscopic+Videos&btnG=)]

* **EPC++**: *"Every pixel counts++: Joint learning of geometry and motion with 3d holistic understanding"*, Luo et al., *TPAMI, 2019*. [[Paper](https://ieeexplore.ieee.org/iel7/34/9185119/08769907.pdf)] [[Code](https://github.com/chenxuluo/EPC)]  [[Bibtex](./bibliography/EPC++.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Every+pixel+counts++:+Joint+learning+of+geometry+and+motion+with+3d+holistic+understanding&btnG=)]

* **SENSE**: *"Sense: A shared encoder network for scene-flow estimation"*, Jiang et al., *ICCV, 2019*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jiang_SENSE_A_Shared_Encoder_Network_for_Scene-Flow_Estimation_ICCV_2019_paper.pdf)] [[Code](https://github.com/NVlabs/SENSE)]  [[Bibtex](./bibliography/SENSE.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Sense:+A+shared+encoder+network+for+scene-flow+estimation&btnG=)]

* **StereoExpansion**: *"Upgrading Optical Flow to 3D Scene Flow through Optical Expansion"*, Yang et al., *ICCV, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Upgrading_Optical_Flow_to_3D_Scene_Flow_Through_Optical_Expansion_CVPR_2020_paper.pdf)] [[Code](https://github.com/gengshan-y/expansion)]  [[Bibtex](./bibliography/StereoExpansion.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Upgrading+Optical+Flow+to+3D+Scene+Flow+through+Optical+Expansion&btnG=)]


* **DWARF**: *"Learning end-to-end scene flow by distilling single tasks knowledge"*, Aleotti et al., *AAAI, 2020*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6613/6467)] [[Code](https://github.com/FilippoAleotti/DWARF-Tensorflow)]  [[Bibtex](./bibliography/DWARF.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+end-to-end+scene+flow+by+distilling+single+tasks+knowledge&btnG=)]

* **SceneFlowFields++**: *"SceneFlowFields++: Multi-frame matching, visibility prediction, and robust interpolation for scene flow estimation"*, Schuster et al., *IJCV, 2020*. [[Paper](https://arxiv.org/pdf/1902.10099.pdf)] [[Bibtex](./bibliography/SceneFlowFields++.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=SceneFlowFields++:+Multi-frame+matching,+visibility+prediction,+and+robust+interpolation+for+scene+flow+estimation&btnG=)]

* **Effiscene**: *"Effiscene: Efficient per-pixel rigidity inference for unsupervised joint learning of optical flow, depth, camera pose and motion segmentation"*, Jiao et al., *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Jiao_EffiScene_Efficient_Per-Pixel_Rigidity_Inference_for_Unsupervised_Joint_Learning_of_CVPR_2021_paper.pdf)] [[Bibtex](./bibliography/Effiscene.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Effiscene:+Efficient+per-pixel+rigidity+inference+for+unsupervised+joint+learning+of+optical+flow,+depth,+camera+pose+and+motion+segmentation&btnG=)]

* **RAFT-3D**: *"RAFT-3D: Scene Flow using Rigid-Motion Embeddings"*, Teed et al., *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Teed_RAFT-3D_Scene_Flow_Using_Rigid-Motion_Embeddings_CVPR_2021_paper.pdf)] [[Code](https://github.com/princeton-vl/RAFT-3D)] [[Bibtex](./bibliography/RAFT-3D.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=RAFT-3D%3A+Scene+Flow+using+Rigid-Motion+Embeddings&btnG=)]

* **RigidMask**: *"Learning to Segment Rigid Motions from Two Frames"*, Yang et al., *CVPR, 2021*. [[Paper](http://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Learning_To_Segment_Rigid_Motions_From_Two_Frames_CVPR_2021_paper.pdf)] [[Code](https://github.com/gengshan-y/rigidmask)] [[Bibtex](./bibliography/RigidMask.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+to+Segment+Rigid+Motions+from+Two+Frames&btnG=)]

* **Self-superflow**: *"Self-superflow: self-supervised scene flow prediction in stereo sequences"*, Bendig et al., *ICIP, 2022*. [[Paper](https://ieeexplore.ieee.org/iel7/9897158/9897159/09897832.pdf)]  [[Bibtex](./bibliography/Self-superflow.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=+Self-SuperFlow%3A+Self-supervised+Scene+Flow+Prediction+in+Stereo+Sequences&btnG=)]

* **CamLiFlow**: *"Learning optical flow and scene flow with bidirectional camera-lidar fusion"*, Liu et al., *TPAMI, 2023*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10310261)] [[Code](https://github.com/MCG-NJU/CamLiFlow)] [[Bibtex](./bibliography/CamLiFlow.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+optical+flow+and+scene+flow+with+bidirectional+camera-lidar+fusion&btnG=)]

* **M-FUSE**: *"M-fuse: Multi-frame fusion for scene flow estimation"*, Mehl et al., *WACV, 2023*. [[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Mehl_M-FUSE_Multi-Frame_Fusion_for_Scene_Flow_Estimation_WACV_2023_paper.pdf)] [[Code](https://github.com/cv-stuttgart/M-FUSE)] [[Bibtex](./bibliography/M-FUSE.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=M-fuse:+Multi-frame+fusion+for+scene+flow+estimation&btnG=)]

* **OpticalExpansion**: *"Learning Optical Expansion from Scale Matching"*, Ling et al., *CVPR, 2023*. [[Paper](http://openaccess.thecvf.com/content/CVPR2023/papers/Ling_Learning_Optical_Expansion_From_Scale_Matching_CVPR_2023_paper.pdf)] [[Bibtex](./bibliography/OpticalExpansion.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+Optical+Expansion+from+Scale+Matching&btnG=)]



</details>

</details>

<details open id="multi-modal">
<summary style="font-size: larger; font-weight: bold;">Beyond Visual Spectrum Deep Stereo Architectures</summary><ul>


<details open>
<summary style="font-size: larger; font-weight: bold;">Depth-Guided Sensor Stereo Networks</summary>

* **LidarStereoFusion**: *"High-precision depth estimation with the 3d lidar and stereo fusion"*, Park et al., *ICRA, 2018*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461048)]  [[Bibtex](./bibliography/LidarStereoFusion.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=High-precision+depth+estimation+with+the+3d+lidar+and+stereo+fusion&btnG=)]

* **GSD**: *"Guided stereo matching"*, Poggi et al., *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Poggi_Guided_Stereo_Matching_CVPR_2019_paper.pdf)] [[Code](https://github.com/mattpoggi/guided-stereo)] [[Bibtex](./bibliography/GSD.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Guided+stereo+matching&btnG=)]

* **LidarStereoNet**: *"Noise-Aware Unsupervised Deep Lidar-Stereo Fusion"*, Cheng et al., *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cheng_Noise-Aware_Unsupervised_Deep_Lidar-Stereo_Fusion_CVPR_2019_paper.pdf)] [[Code](https://github.com/XuelianCheng/LidarStereoNet)] [[Bibtex](./bibliography/LidarStereoNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Noise-Aware+Unsupervised+Deep+Lidar-Stereo+Fusion&btnG=)]

* **Stereo-LiDAR-CCVNorm**: *"3d lidar and stereo fusion using stereo matching network with conditional cost volume normalization"*, Wang et al., *IROS, 2019*. [[Paper](https://ieeexplore.ieee.org/abstract/document/8968170?casa_token=ejW3T3WNT8IAAAAA:LN2rFtefaSnrUWMzZFIE8LJESFM6WbY3xQwjM-jzSo0fGHTBTqVjPa1f2pYrqbz1Oe173gUZ)] [[Code](https://github.com/zswang666/Stereo-LiDAR-CCVNorm)] [[Bibtex](./bibliography/LidarStereoNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=3D+LiDAR+and+Stereo+Fusion+using+Stereo+Matching+Network+with+Conditional+Cost+Volume+Normalization&btnG=)]

* **Pseudo-LiDAR++**: *"Pseudo-LiDAR++: Accurate Depth for 3D Object Detection in Autonomous Driving"*, You et al., *ICLR, 2020*. [[Paper](https://arxiv.org/pdf/1906.06310.pdf)] [[Code](https://github.com/mileyan/Pseudo_Lidar_V2)] [[Bibtex](./bibliography/Pseudo-LiDAR++.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Pseudo-LiDAR++:+Accurate+Depth+for+3D+Object+Detection+in+Autonomous+Driving&btnG=)]

* **Listereo**: *"Listereo: Generate dense depth maps from lidar and stereo imagery"*, Zhang et al., *ICRA, 2020*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9196628)] [[Bibtex](./bibliography/Listereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Listereo%3A+Generate+dense+depth+maps+from+lidar+and+stereo+imagery&btnG=)]

* **S<sup>3**: *"S<sup>3</sup>: Learnable sparse signal superdensity for guided depth estimation"*, Huang et al., CVPR, 2021. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_S3_Learnable_Sparse_Signal_Superdensity_for_Guided_Depth_Estimation_CVPR_2021_paper.pdf)] [[Bibtex](./bibliography/S3.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=+Learnable+sparse+signal+superdensity+for+guided+depth+estimation&btnG=)]

* **LSMD-Net**: *"LSMD-Net: LiDAR-Stereo Fusion with Mixture Density Network for Depth Sensing"*, Yin et al., *ACCV, 2022*. [[Paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Yin_LSMD-Net_LiDAR-Stereo_Fusion_with_Mixture_Density_Network_for_Depth_Sensing_ACCV_2022_paper.pdf)] [[Bibtex](./bibliography/LSMD-Net.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=LSMD-Net:+LiDAR-Stereo+Fusion+with+Mixture+Density+Network+for+Depth+Sensing&btnG=)]

* **CamLiFlow**: *"Learning optical flow and scene flow with bidirectional camera-lidar fusion"*, *TPAMI, 2023*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10310261)] [[Code](https://github.com/MCG-NJU/CamLiFlow)] [[Bibtex](./bibliography/CamLiFlow.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+optical+flow+and+scene+flow+with+bidirectional+camera-lidar+fusion&btnG=)]


* **Active Disparity Sampling**: *"Active Disparity Sampling for Stereo Matching With Adjoint Network"*, Zhang et al., *TIP, 2023*. [[Paper](https://ieeexplore.ieee.org/iel7/83/4358840/10367813.pdf?casa_token=LTr0bj333I8AAAAA:VbxzhxvyA--lpHpIUfFhMkHADyz8v4kIq6jxdQFYgi_ruTy2ahOIZk13YjuE0ynwkHfUnxOV)] [[Bibtex](./bibliography/ActiveDisparitySampling.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Active+Disparity+Sampling+for+Stereo+Matching+With+Adjoint+Network&btnG=)]


* **VPP**: *"Active Stereo Without Pattern Projector"*, Bartolomei et al., *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Bartolomei_Active_Stereo_Without_Pattern_Projector_ICCV_2023_paper.pdf)] [[Code](https://vppstereo.github.io/)] [[Bibtex](./bibliography/VPP.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Active+Stereo+Without+Pattern+Projector&btnG=)]

* **SDG-Depth**: *"Stereo-LiDAR Depth Estimation with Deformable Propagation and Learned Disparity-Depth Conversion"*, Li et al., *ICRA, 2024*. [[Paper](https://arxiv.org/pdf/2404.07545.pdf)] [[Code](https://github.com/SJTU-ViSYS/SDG-Depth)] [[Bibtex](./bibliography/SDG-Depth.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Stereo-LiDAR+Depth+Estimation+with+Deformable+Propagation+and+Learned+Disparity-Depth+Conversion&btnG=)]

* **VPP-Extended**: *"Stereo-Depth Fusion through Virtual Pattern Projection"*, Bartolomei et al., *arXiv, 2024*. [[Paper](https://arxiv.org/pdf/2406.04345)] [[Code](https://github.com/bartn8/vppstereo)] [[WebPage](https://vppstereo.github.io/extension.html)] [[Bibtex](./bibliography/VPP-Extended.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Stereo-Depth+Fusion+through+Virtual+Pattern+Projection&btnG=)]


* **D3RoMa**: *"D3RoMa: Disparity Diffusion-based Depth Sensing for Material-Agnostic Robotic Manipulation"*, Wei et al., *CoRL, 2024*. [[Paper](https://arxiv.org/pdf/2405.05648)] [[WebPage](https://pku-epic.github.io/D3RoMa/)]  [[Bibtex](./bibliography/D3RoMa.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=D3RoMa%3A+Disparity+Diffusion-based+Depth+Sensing+for+Material-Agnostic+Robotic+Manipulation&btnG=)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Pattern Projection-Based Stereo Networks</summary>

* **ActiveStereoNet**: *"ActiveStereoNet: End-to-End Self-Supervised Learning for Active Stereo Systems"*, Zhang et al., *ECCV, 2018*. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yinda_Zhang_Active_Stereo_Net_ECCV_2018_paper.pdf)] [[Bibtex](./bibliography/ActiveStereoNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=ActiveStereoNet:+End-to-End+Self-Supervised+Learning+for+Active+Stereo+Systems&btnG=)]

* **Polka Lines**: *"Polka Lines: Learning Structured Illumination and Reconstruction for Active Stereo"*, Baek et al., *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Baek_Polka_Lines_Learning_Structured_Illumination_and_Reconstruction_for_Active_Stereo_CVPR_2021_paper.pdf)] [[Bibtex](./bibliography/Polka.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Polka+Lines:+Learning+Structured+Illumination+and+Reconstruction+for+Active+Stereo&btnG=)]

* **Activezero**: *"Activezero: Mixed domain learning for active stereovision with zero annotation"*, Liu et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_ActiveZero_Mixed_Domain_Learning_for_Active_Stereovision_With_Zero_Annotation_CVPR_2022_paper.pdf)] [[Code](https://github.com/haosulab/ActiveZero/tree/master)] [[Bibtex](./bibliography/Activezero.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Activezero:+Mixed+domain+learning+for+active+stereovision+with+zero+annotation&btnG=)]

* **MonoStereoFusion**: *"Depth Estimation by Combining Binocular Stereo and Monocular Structured-Light"*, Xu et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Depth_Estimation_by_Combining_Binocular_Stereo_and_Monocular_Structured-Light_CVPR_2022_paper.pdf)] [[Code](https://github.com/YuhuaXu/MonoStereoFusion)] [[Bibtex](./bibliography/MonoStereoFusion.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Depth+Estimation+by+Combining+Binocular+Stereo+and+Monocular+Structured-Light&btnG=)]


* **Activezero++**: *"Activezero++: Mixed domain learning stereo and confidence-based depth completion with zero annotation"*, Chen et al., *TPAMI, 2023*. [[Paper](https://ieeexplore.ieee.org/abstract/document/10219021)] [[Bibtex](./bibliography/Activezero++.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Activezero++:+Mixed+domain+learning+stereo+and+confidence-based+depth+completion+with+zero+annotation&btnG=)]

* **ASGrasp**: *"ASGrasp: Generalizable Transparent Object Reconstruction and 6-DoF Grasp Detection from RGB-D Active Stereo Camera"*, Shi et al., *ICRA, 2024*. [[Paper](https://arxiv.org/pdf/2405.05648)] [[WebPage](https://pku-epic.github.io/ASGrasp)] [[Bibtex](./bibliography/ASGrasp.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=ASGrasp%3A+Generalizable+Transparent+Object+Reconstruction+and+Grasping+from+RGB-D+Active+Stereo+Camera&btnG=)]


</details>


<details open>
<summary style="font-size: larger; font-weight: bold;">Cross-Spectral Stereo Networks</summary>

* **CS-Stereo**: *"Deep material-aware cross-spectral stereo matching"*, Zhi et al., *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhi_Deep_Material-Aware_Cross-Spectral_CVPR_2018_paper.pdf)] [[Code](https://github.com/tiancheng-zhi/cs-stereo)] [[Bibtex](./bibliography/CS-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Deep+material-aware+cross-spectral+stereo+matching&btnG=)]

* **UCSS**: *"Unsupervised cross-spectral stereo matching by learning to synthesize"*, Liang et al., *AAAI, 2019*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/4894/4767)] [[Code - Unofficial](https://github.com/rish-av/cross_spectral_stereo)] [[Bibtex](./bibliography/UCSS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Unsupervised+cross-spectral+stereo+matching+by+learning+to+synthesize&btnG=)]

* **SS-MCE**: *"There and back again: Self-supervised multispectral correspondence estimation"*, Walters et al., *ICRA, 2021*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9561621)]  [[Bibtex](./bibliography/SS-MCE.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=There+and+back+again:+Self-supervised+multispectral+correspondence+estimation&btnG=)]

* **RGB-MS**: *"RGB-Multispectral matching: Dataset, learning methodology, evaluation"*, Tosi et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tosi_RGB-Multispectral_Matching_Dataset_Learning_Methodology_Evaluation_CVPR_2022_paper.pdf)] [[Code](https://cvlab-unibo.github.io/rgb-ms-web/)] [[Bibtex](./bibliography/RGB-MS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=RGB-Multispectral+matching:+Dataset,+learning+methodology,+evaluation&btnG=)]

* **DPS-Net**: *"DPS-Net: Deep Polarimetric Stereo Depth Estimation"*, Tian et al., *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Tian_DPS-Net_Deep_Polarimetric_Stereo_Depth_Estimation_ICCV_2023_paper.pdf)] [[Code](https://github.com/Ethereal-Tian/DPS-Net)] [[Bibtex](./bibliography/DPS-Net.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=DPS-Net:+Deep+Polarimetric+Stereo+Depth+Estimation&btnG=)]

* **CrossSP**: *"Unsupervised Cross-Spectrum Depth Estimation by Visible-Light and Thermal Cameras"*, Guo et al., *T-ITS, 2023*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10146199)] [[Code](https://github.com/whitecrow1027/CrossSP_Depth)] [[Bibtex](./bibliography/CrossSP.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Unsupervised+Cross-Spectrum+Depth+Estimation+by+Visible-Light+and+Thermal+Cameras&btnG=)]

* **Gated-RCCB**: *"Cross-spectral Gated-RGB Stereo Depth Estimation"*, Brucker et al., *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2405.12759)] [[WebPage](https://light.princeton.edu/gatedrccbstereo/)] [[Bibtex](./bibliography/Gated-RCCB.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Cross-spectral+Gated-RGB+Stereo+Depth+Estimation&btnG=)]

</details>


<details open>
<summary style="font-size: larger; font-weight: bold;">Event Stereo Networks</summary>

* **Event-IntensityStereo**: *"Event-Intensity Stereo: Estimating Depth by the Best of Both Worlds"*, Mostafavi et al., *ICCV, 2021*. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Mostafavi_Event-Intensity_Stereo_Estimating_Depth_by_the_Best_of_Both_Worlds_ICCV_2021_paper.pdf)] [[Code](https://github.com/yonseivnl/se-cff)] [[Bibtex](./bibliography/Event-IntensityStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Event-Intensity+Stereo:+Estimating+Depth+by+the+Best+of+Both+Worlds&btnG=)]

* **SE-CFF**: *"Stereo Depth From Events Cameras: Concentrate and Focus on the Future"*, Nam et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Nam_Stereo_Depth_From_Events_Cameras_Concentrate_and_Focus_on_the_CVPR_2022_paper.pdf)] [[Code](https://github.com/yonseivnl/se-cff)] [[Bibtex](./bibliography/SE-CFF.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Stereo+Depth+From+Events+Cameras:+Concentrate+and+Focus+on+the+Future&btnG=)]

* **SCSNet**: *"Selection and Cross Similarity for Event-Image Deep Stereo"*, Cho et al., *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920467.pdf)] [[Code](https://github.com/Chohoonhee/SCSNet)] [[Bibtex](./bibliography/SCSNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Selection+and+Cross+Similarity+for+Event-Image+Deep+Stereo&btnG=)]

* **DTC-SPADE**: *"Discrete Time Convolution for Fast Event-Based Stereo"*, Zhang et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Discrete_Time_Convolution_for_Fast_Event-Based_Stereo_CVPR_2022_paper.pdf)] [[Bibtex](./bibliography/DTC-SPADE.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Discrete+Time+Convolution+for+Fast+Event-Based+Stereo&btnG=)]

* **EFS**: *"Event-image fusion stereo using cross-modality feature propagation"*, Cho et al., *AAAI, 2022*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/19923/19682)] [[Bibtex](./bibliography/EFS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Event-Image+Fusion+Stereo+Using+Cross-Modality+Feature+Propagation&btnG=)]

* **ADES**: *"Learning Adaptive Dense Event Stereo From the Image Domain"*, Cho et al., *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Cho_Learning_Adaptive_Dense_Event_Stereo_From_the_Image_Domain_CVPR_2023_paper.pdf)] [[Bibtex](./bibliography/ADES.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+Adaptive+Dense+Event+Stereo+From+the+Image+Domain&btnG=)]

* **SAFE**: *"Depth From Asymmetric Frame-Event Stereo: A Divide-and-Conquer Approach"*, Chen et al., *WACV, 2024*. [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Chen_Depth_From_Asymmetric_Frame-Event_Stereo_A_Divide-and-Conquer_Approach_WACV_2024_paper.pdf)] [[Bibtex](./bibliography/SAFE.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Depth+From+Asymmetric+Frame-Event+Stereo:+A+Divide-and-Conquer+Approach&btnG=)]

* **TemporalEventStereo**: *"Temporal Event Stereo via Joint Learning with Stereoscopic Flow"*, Cho et al., *ECCV, 2024*. [[Paper](https://arxiv.org/pdf/2407.10831)] [[Code](https://github.com/mickeykang16/TemporalEventStereo)] [[Bibtex](./bibliography/TemporalEventStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Temporal+Event+Stereo+via+Joint+Learning+with+Stereoscopic+Flow&btnG=)]

* **EventVPPStereo**: *"LiDAR-Event Stereo Fusion with Hallucinations"*, Bartolomei et al., *ECCV, 2024*. [[Paper](https://arxiv.org/pdf/2408.04633)] [[WebPage](https://eventvppstereo.github.io/)] [[Code](https://github.com/bartn8/eventvppstereo/)] [[Bibtex](./bibliography/EventVPPStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=LiDAR-Event+Stereo+Fusion+with+Hallucinations&btnG=)]

* **ZEST**: "*Zero-Shot Event-Intensity Asymmetric Stereo via Visual Prompting from Image Domain*", Lou et al., *NeurIPS, 2024*. [[Paper](https://openreview.net/pdf?id=E3ZMsqdO0D)] [[Code](https://github.com/HYLZ-2019/ZEST)] [[Bibtex](./bibliography/ZEST.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Zero-Shot+Event-Intensity+Asymmetric+Stereo+via+Visual+Prompting+from+Image+Domain&btnG=)] 

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Gated Stereo Networks</summary>

* **GatedStereo**: *"Gated Stereo: Joint Depth Estimation from Gated and Wide-Baseline Active Stereo Cues"*, Walz et al., *CVPR, 2023*. [[Paper](http://openaccess.thecvf.com/content/CVPR2023/papers/Walz_Gated_Stereo_Joint_Depth_Estimation_From_Gated_and_Wide-Baseline_Active_CVPR_2023_paper.pdf)] [[WebPage](https://light.princeton.edu/publication/gatedstereo/)] [[Bibtex](./bibliography/Gated.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Gated+Stereo:+Joint+Depth+Estimation+from+Gated+and+Wide-Baseline+Active+Stereo+Cues&btnG=)]

* **Gated-RCCB**: *"Cross-spectral Gated-RGB Stereo Depth Estimation"*, Brucker et al., *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2405.12759)] [[WebPage](https://light.princeton.edu/gatedrccbstereo/)] [[Bibtex](./bibliography/Gated-RCCB.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Cross-spectral+Gated-RGB+Stereo+Depth+Estimation&btnG=)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Stereo Networks with Echoes </summary>

* **StereoEchoes**: *"Stereo Depth Estimation with Echoes"*, Zhang et al., *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870489.pdf)] [[Bibtex](./bibliography/StereoEchoes.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Stereo+Depth+Estimation+with+Echoes&btnG=)]


</details>

</ul>
</details>

### Architectural Analysis

* **OpenStereo**: *"OpenStereo: A Comprehensive Benchmark for Stereo Matching and Strong Baseline"*, Xianda et al., *arXiv, 2023*  [[Paper](https://arxiv.org/pdf/2312.00343.pdf)] [[Code](https://github.com/XiandaGuo/OpenStereo)] [[Bibtex](./bibliography/OpenStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=OpenStereo:+A+Comprehensive+Benchmark+for+Stereo+Matching+and+Strong+Baseline&btnG=)]

* *"Exploring the Usage of Pre-trained Features for Stereo Matching"*, Zhang et al., *IJCV, 2024*  [[Paper](https://link.springer.com/article/10.1007/s11263-024-02090-y)] [[Bibtex](./bibliography/ExploringFeatures.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Exploring+the+Usage+of+Pre-trained+Features+for+Stereo+Matching&btnG=)]

### Challenges & Solutions

<details open id="over-smoothing">
<summary style="font-size: larger; font-weight: bold;">Addressing the Over-Smoothing Issue</summary>

* **SM-CDE**: *"On the over-smoothing problem of cnn based disparity estimation"*, Chen et al., *ICCV, 2019*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_On_the_Over-Smoothing_Problem_of_CNN_Based_Disparity_Estimation_ICCV_2019_paper.html)] [[Bibtex](./bibliography/SM-CDE.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=On+the+over-smoothing+problem+of+cnn+based+disparity+estimation&btnG=)] 

* **AcfNet**: *"Adaptive Unimodal Cost Volume Filtering for Deep Stereo Matching"*, Zhang et al., *AAAI, 2020*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6991/6845)] [[Code](https://github.com/youmi-zym/AcfNet)] [[Bibtex](./bibliography/WaveletStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Adaptive+Unimodal+Cost+Volume+Filtering+for+Deep+Stereo+Matching&btnG=)]


* **CDN**: *"Wasserstein Distances for Stereo Disparity Estimation"*, Garg et al., *NeurIPS, 2020*. [[Paper](https://papers.nips.cc/paper/2020/file/fe7ecc4de28b2c83c016b5c6c2acd826-Paper.pdf)] [[Code](https://github.com/Div99/W-Stereo-Disp)] [[Bibtex](./bibliography/CDN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Wasserstein+Distances+for+Stereo+Disparity+Estimation&btnG=)] 

* **SMD-Nets**: *"SMD-Nets: Stereo Mixture Density Networks"*, Tosi et al.,  *CVPR, 2021*. [[Paper](http://www.cvlibs.net/publications/Tosi2021CVPR.pdf)] [[Code](https://github.com/fabiotosi92/SMD-Nets)] [[Bibtex](./bibliography/SMD-Nets.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=SMD-Nets:+Stereo+Mixture+Density+Networks&btnG=)] 

* **NDR**: "*Neural disparity refinement for arbitrary resolution stereo*", Aleotti et al., *3DV, 2021*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9665913?casa_token=3rm4WpqLb_QAAAAA:5Sa0RO547j8LsaEYUeppzB33gZJg5Y3tfiPVwM9rzs9MEAuoHSta0Kdw3Cm9NrtfOOdFkIwp)] [[Website](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Neural+disparity+refinement+for+arbitrary+resolution+stereo&btnG=)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Neural+disparity+refinement+for+arbitrary+resolution+stereo&btnG=)]  

* **LaC**: "*Local similarity pattern and cost self-reassembling for deep stereo matching networks*", Liu et al., *AAAI, 2022*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20056/19815)] [[Code](https://github.com/SpadeLiu/Lac-GwcNet)] [[Bibtex](./bibliography/LaC.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Local+similarity+pattern+and+cost+self-reassembling+for+deep+stereo+matching+networks&btnG=)] 

* **ADL**: "*Adaptive Multi-Modal Cross-Entropy Loss for Stereo Matching*", Xu et al., *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2306.15612.pdf)] [[Code](https://github.com/xxxupeng/ADL)] [[Bibtex](./bibliography/ADL.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Adaptive+Multi-Modal+Cross-Entropy+Loss+for+Stereo+Matching&btnG=)] 

* **NDR v2**: "*Neural disparity refinement*", Tosi et al., *TPAMI, 2024*. [[Paper](https://ieeexplore.ieee.org/iel8/34/4359286/10552115.pdf)] [[Website](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR_v2.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Neural+disparity+refinement&btnG=)]  

* **Stereo Risk**: "*Stereo Risk: A Continuous Modeling Approach to Stereo Matching*", Liu et al., *ICML, 2024*. [[Paper](https://arxiv.org/pdf/2407.03152)]  [[Bibtex](./bibliography/StereoRisk.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&scisbd=1&as_sdt=2005&sciodt=0%2C5&cites=17071509949921577102&scipsc=&q=Stereo+Risk%3A+A+Continuous+Modeling+Approach+to+Stereo+Matching&btnG=)] 


</details>


<details open id="missing-gt">
<summary style="font-size: larger; font-weight: bold;">Missing Ground Truth Depth</summary><ul>

<details open>
<summary style="font-size: larger; font-weight: bold;">Self-Supervised</summary>

* :triangular_flag_on_post: **MonoDepth/StereoDepth**: *"Unsupervised monocular depth estimation with left-right consistency"*, Godard et al., *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Godard_Unsupervised_Monocular_Depth_CVPR_2017_paper.pdf)] [[Code](https://github.com/mrharicot/monodepth)] [[Bibtex](./bibliography/MonoDepth.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Unsupervised+monocular+depth+estimation+with+left-right+consistency&btnG=)]

* **USM**: *"Unsupervised learning of stereo matching"*, Zhou et al., *ICCV, 2017*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhou_Unsupervised_Learning_of_ICCV_2017_paper.pdf)] [[Bibtex](./bibliography/Flow2Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Unsupervised+learning+of+stereo+matching&btnG=)]

* **OASM-Net**: *"Occlusion aware stereo matching via cooperative unsupervised learning"*, Li et al., *ACCV, 2018*. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-20876-9_13)] [[Bibtex](./bibliography/OASM-Net.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Occlusion+aware+stereo+matching+via+cooperative+unsupervised+learning&btnG=)]

* **UnOS**: *"UnOS: Unified Unsupervised Optical-Flow and Stereo-Depth Estimation by Watching Videos"*, Wang et al., *CVPR, 2019*. [[Paper](https://ieeexplore.ieee.org/iel7/34/9185119/08769907.pdf)] [[Code](https://github.com/baidu-research/UnDepthflow)]  [[Bibtex](./bibliography/UnOS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=UnOS:+Unified+Unsupervised+Optical-Flow+and+Stereo-Depth+Estimation+by+Watching+Videos&btnG=)]

* **BridgeDepthFlow**: *"Bridging Stereo Matching and Optical Flow via Spatiotemporal Correspondence"*, *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lai_Bridging_Stereo_Matching_and_Optical_Flow_via_Spatiotemporal_Correspondence_CVPR_2019_paper.pdf)] [[Code](https://github.com/lelimite4444/BridgeDepthFlow)] [[Bibtex](./bibliography/BridgeDepthFlow.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Bridging+Stereo+Matching+and+Optical+Flow+via+Spatiotemporal+Correspondence&btnG=)]

* **Correspondence Consistency**: *"Unsupervised stereo matching using confidential correspondence consistency"*, Joung et al., *T-ITS, 2019*. [[Paper](https://ieeexplore.ieee.org/iel7/6979/4358928/08721668.pdf?casa_token=YfFwIZKeGJ4AAAAA:LCkTQefrcYIOljN6Yyc55dxXCUthgkrnmJLwu7gSjD_Cd65HZ_EOpGDJq49V5GylbgxtGKG4)] [[Bibtex](./bibliography/Correspondence_Consistency.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Unsupervised+stereo+matching+using+confidential+correspondence+consistency&btnG=)]

* **Flow2Stereo**: *"Flow2Stereo: Effective Self-Supervised Learning of Optical Flow and Stereo Matching"*, Liu et al., *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Flow2Stereo_Effective_Self-Supervised_Learning_of_Optical_Flow_and_Stereo_Matching_CVPR_2020_paper.pdf)] [[Code](https://github.com/ppliuboy/Flow2Stereo)] [[Bibtex](./bibliography/Flow2Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Flow2Stereo:+Effective+Self-Supervised+Learning+of+Optical+Flow+and+Stereo+Matching&btnG=)]

* **PASMNet**: *"Parallax attention for unsupervised stereo correspondence learning"*, Wang et al., *TPAMI, 2020*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9206116)] [[Code](https://github.com/The-Learning-And-Vision-Atelier-LAVA/PAM)] [[Bibtex](./bibliography/PASMNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Parallax+attention+for+unsupervised+stereo+correspondence+learning&btnG=)]

* **MultiscopicVision**: *"Stereo matching by self-supervision of multiscopic vision"*, Yuan et al., *IROS, 2021*. [[Paper](https://cqf.io/papers/Stereo_Matching_Multiscopic_Vision_IROS2021.pdf)] [[WebPage](https://sites.google.com/view/multiscopic)] [[Bibtex](./bibliography/MultiscopicVision.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Stereo+matching+by+self-supervision+of+multiscopic+vision&btnG=)]

* **Feature-Level Collaboration**: *"Feature-Level Collaboration: Joint Unsupervised Learning of Optical Flow, Stereo Depth and Camera Motion"*, Chi et al., *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chi_Feature-Level_Collaboration_Joint_Unsupervised_Learning_of_Optical_Flow_Stereo_Depth_CVPR_2021_paper.pdf)] [[Bibtex](./bibliography/Feature-Level_Collaboration.txt)]

* **Occlusion-Aware Stereo**: *"Unsupervised Occlusion-Aware Stereo Matching With Directed Disparity Smoothing"*, Li et al., *T-ITS, 2022*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9404889)] [[Bibtex](./bibliography/Occlusion-Aware_Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Unsupervised+Occlusion-Aware+Stereo+Matching+With+Directed+Disparity+Smoothing&btnG=)]

</details>
<details open >
<summary style="font-size: larger; font-weight: bold;">Cross-Framework/Proxy Supervision</summary>

* **Reversing-Stereo**: *"Reversing the cycle: self-supervised deep stereo through enhanced monocular distillation"*, Aleotti et al., *ECCV, 2020*. [[Paper](https://arxiv.org/pdf/2008.07130.pdf)] [[Code](https://github.com/FilippoAleotti/Reversing)] [[Bibtex](./bibliography/Reversing-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Reversing+the+cycle:+self-supervised+deep+stereo+through+enhanced+monocular+distillation&btnG=)]

* **Revealing-Stereo**: *"Revealing the Reciprocal Relations between Self-Supervised Stereo and Monocular Depth Estimation"*, Chen et al., *ICCV, 2021*. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Revealing_the_Reciprocal_Relations_Between_Self-Supervised_Stereo_and_Monocular_Depth_ICCV_2021_paper.pdf)]  [[Bibtex](./bibliography/Revealing-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Revealing+the+Reciprocal+Relations+between+Self-Supervised+Stereo+and+Monocular+Depth+Estimation&btnG=)]

* **TiO-Depth**: *"Two-in-one depth: Bridging the gap between monocular and binocular self-supervised depth estimation"*, Zhou et al., *ICCV, 2023*. [[Paper](http://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Two-in-One_Depth_Bridging_the_Gap_Between_Monocular_and_Binocular_Self-Supervised_ICCV_2023_paper.pdf)] [[Code](https://github.com/ZM-Zhou/TiO-Depth_pytorch)] [[Bibtex](./bibliography/Two-in-One.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Two-in-one+depth:+Bridging+the+gap+between+monocular+and+binocular+self-supervised+depth+estimation&btnG=)]

* **NeRF-Supervised Stereo**: "*NeRF-Supervised Deep Stereo*", Tosi et al., *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tosi_NeRF-Supervised_Deep_Stereo_CVPR_2023_paper.pdf)] [[Website](https://nerfstereo.github.io/)] [[Code](https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo)] [[Bibtex](./bibliography/NS-Stereo.txt)] 

* **SAG**: "*Self-Assessed Generation: Trustworthy Label Generation for Optical Flow and Stereo Matching in Real-world*", Ling et al., *arXiv, 2024*. [[Paper](https://arxiv.org/pdf/2410.10453)] [[Code](https://github.com/HanLingsgjk/UnifiedGeneralization)] [[Bibtex](./bibliography/SAG.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Self-Assessed+Generation%3A+Trustworthy+Label+Generation+for+Optical+Flow+and+Stereo+Matching+in+Real-world+&btnG=)] 

</details>




</ul>
</details>

<details open id="domain-shift">
<summary style="font-size: larger; font-weight: bold;">Domain Shift</summary><ul>

<details open>
<summary style="font-size: larger; font-weight: bold;">Zero-shot Generalization</summary><ul>

<details open>
<summary style="font-size: larger; font-weight: bold;">Domain-Agnostic Feature Modeling</summary>

* :triangular_flag_on_post: **DSM-Net**: "*Domain-invariant Stereo Matching Networks*", Zhang et al., *ECCV, 2020*. [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470409.pdf)] [[Code](https://github.com/feihuzhang/DSMNet)] [[Bibtex](./bibliography/DSM-Net.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Domain-invariant+Stereo+Matching+Networks&btnG=)] 


* **FCStereo**: "*Revisiting Domain Generalized Stereo Matching Networks From a Feature Consistency Perspective*", Zhang et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Revisiting_Domain_Generalized_Stereo_Matching_Networks_From_a_Feature_Consistency_CVPR_2022_paper.pdf)] [[Code](https://github.com/jiaw-z/FCStereo)] [[Bibtex](./bibliography/FCStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Revisiting+Domain+Generalized+Stereo+Matching+Networks+From+a+Feature+Consistency+Perspective&btnG=)] 

* **GraftNet**: "*GraftNet: Towards Domain Generalized Stereo Matching With a Broad-Spectrum and Task-Oriented Feature*", Liu et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_GraftNet_Towards_Domain_Generalized_Stereo_Matching_With_a_Broad-Spectrum_and_CVPR_2022_paper.pdf)] [[Code](https://github.com/SpadeLiu/Graft-PSMNet)] [[Bibtex](./bibliography/GraftNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=GraftNet%3A+Towards+Domain+Generalized+Stereo+Matching+With+a+Broad-Spectrum+and+Task-Oriented+Feature&btnG=)] 

* **ITSA**: "*ITSA: An Information-Theoretic Approach to Automatic Shortcut Avoidance and Domain Generalization in Stereo Matching Networks*", Chuah et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chuah_ITSA_An_Information-Theoretic_Approach_to_Automatic_Shortcut_Avoidance_and_Domain_CVPR_2022_paper.pdf)] [[Code](https://github.com/waychin-weiqin/ITSA)] [[Bibtex](./bibliography/ITSA.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=ITSA%3A+An+Information-Theoretic+Approach+to+Automatic+Shortcut+Avoidance+and+Domain+Generalization+in+Stereo+Matching+Network&btnG=)] 

* **HVT**: "*Domain Generalized Stereo Matching via Hierarchical Visual Transformation*", Chang et al., *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chang_Domain_Generalized_Stereo_Matching_via_Hierarchical_Visual_Transformation_CVPR_2023_paper.pdf)] [[Code](https://github.com/cty8998/HVT-PSMNet)] [[Bibtex](./bibliography/HVT.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Domain+Generalized+Stereo+Matching+via+Hierarchical+Visual+Transformation&btnG=)] 

* **MRL-Stereo**: "*Masked representation learning for domain generalized stereo matching*", Rao et al., *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Rao_Masked_Representation_Learning_for_Domain_Generalized_Stereo_Matching_CVPR_2023_paper.pdf)] [[Bibtex](./bibliography/MRL-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Masked+representation+learning+for+domain+generalized+stereo+matching&btnG=)] 

* **HODC**: "*Hierarchical Object-Aware Dual-Level Contrastive Learning for Domain Generalized Stereo Matching*", Miao et al., *NeurIPS, 2024*. [[Paper](https://openreview.net/pdf?id=HcqV2bPFKz)] [[Website](https://joshmiao.github.io/HODC/)] [[Bibtex](./bibliography/HODC.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Hierarchical+Object-Aware+Dual-Level+Contrastive+Learning+for+Domain+Generalized+Stereo+Matching&btnG=)] 

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Non-parametric Cost Volumes</summary>

* **MS-Nets**: "*Matching-space Stereo Networks for Cross-domain Generalization*", Cai et al., *3DV, 2020*. [[Paper](https://mordohai.github.io/public/Cai_MatchingSpaceStereo20.pdf)] [[Code](https://github.com/ccj5351/MS-Nets)] [[Bibtex](./bibliography/MS-Nets.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Matching-space+Stereo+Networks+for+Cross-domain+Generalization&btnG=)] 

* **ARStereo**: "*Revisiting Non-Parametric Matching Cost Volumes for Robust and Generalizable Stereo Matching*", Cheng et al., *NeurIPS, 2022*. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/6794f555524c9069e26970a408d353cc-Paper-Conference.pdf)] [[Code](https://github.com/kelkelcheng/AdversariallyRobustStereo)]  [[Bibtex](./bibliography/ARStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Revisiting+Non-Parametric+Matching+Cost+Volumes+for+Robust+and+Generalizable+Stereo+Matching&btnG=)] 


</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Integration of Additional Geometric Cues</summary>

* **NDR**: "*Neural disparity refinement for arbitrary resolution stereo*", Aleotti et al., *3DV, 2021*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9665913?casa_token=3rm4WpqLb_QAAAAA:5Sa0RO547j8LsaEYUeppzB33gZJg5Y3tfiPVwM9rzs9MEAuoHSta0Kdw3Cm9NrtfOOdFkIwp)] [[Website](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Neural+disparity+refinement+for+arbitrary+resolution+stereo&btnG=)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Neural+disparity+refinement+for+arbitrary+resolution+stereo&btnG=)]  


* **EVHS**: "*Expansion of Visual Hints for Improved Generalization in Stereo Matching*", Pilzer et al., *WACV, 2023*. [[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Pilzer_Expansion_of_Visual_Hints_for_Improved_Generalization_in_Stereo_Matching_WACV_2023_paper.pdf)] [[Bibtex](./bibliography/EVHS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Expansion+of+Visual+Hints+for+Improved+Generalization+in+Stereo+Matching&btnG=)] 


* **NDR v2**: "*Neural disparity refinement*", Tosi et al., *TPAMI, 2024*. [[Paper](https://ieeexplore.ieee.org/iel8/34/4359286/10552115.pdf)] [[Website](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR_v2.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Neural+disparity+refinement&btnG=)]  

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Real-World Monocular to Synthetic Stereo Data</summary>

 
* **LSSI**: "*Learning Stereo from Single Images*", Watson et al., *ECCV, 2020*. [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460698.pdf)] [[Code](https://github.com/nianticlabs/stereo-from-mono/)] [[Bibtex](./bibliography/LSSI.txt)]  [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Learning+Stereo+from+Single+Images&btnG=)] 

* **NeRF-Supervised Stereo**: "*NeRF-Supervised Deep Stereo*", Tosi et al., *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tosi_NeRF-Supervised_Deep_Stereo_CVPR_2023_paper.pdf)] [[Website](https://nerfstereo.github.io/)] [[Code](https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo)] [[Bibtex](./bibliography/NS-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=NeRF-Supervised+Deep+Stereo&btnG=)] 

* **SAG**: "*Self-Assessed Generation: Trustworthy Label Generation for Optical Flow and Stereo Matching in Real-world*", Ling et al., *arXiv, 2024*. [[Paper](https://arxiv.org/pdf/2410.10453)] [[Code](https://github.com/HanLingsgjk/UnifiedGeneralization)] [[Bibtex](./bibliography/SAG.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Self-Assessed+Generation%3A+Trustworthy+Label+Generation+for+Optical+Flow+and+Stereo+Matching+in+Real-world+&btnG=)] 

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Knowledge Transfer</summary>

* **DKT-Stereo**: "*Robust Synthetic-to-Real Transfer for Stereo Matching*", Zhang et al., *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2403.07705)] [[Code](https://github.com/jiaw-z/DKT-Stereo)] [[Bibtex](./bibliography/DKT-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Robust+Synthetic-to-Real+Transfer+for+Stereo+Matching&btnG=)] 


</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Monocular-Stereo Integration</summary>

   * **LoS**: *"LoS: Local Structure-guided Stereo Matching"*, Li et al., *CVPR, 2024*. [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_LoS_Local_Structure-Guided_Stereo_Matching_CVPR_2024_paper.pdf)] [[Code](https://github.com/SYSU-SAIL/LoS)] [[Bibtex](./bibliography/LoS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=LoS%3A+Local+Structure-guided+Stereo+Matching&btnG=)]

   * **MonSter**: *"MonSter: Marry Monodepth to Stereo Unleashes Power"*, Cheng et al., *CVPR, 2025*. [[Paper](https://arxiv.org/pdf/2501.08643)] [[Code](https://github.com/Junda24/MonSter)] [[Bibtex](./bibliography/MonSter.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=MonSter%3A+Marry+Monodepth+to+Stereo+Unleashes+Power&btnG=)]

   * **Stereo Anywhere**: *"Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching Even Where Either Stereo or Mono Fail"*, Bartolomei et al., *CVPR, 2025*. [[Paper](http://arxiv.org/abs/2412.04472)] [[Code](https://github.com/bartn8/stereoanywhere/)] [[WebPage](https://stereoanywhere.github.io/)] [[Bibtex](./bibliography/StereoAnywhere.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Stereo+Anywhere%3A+Robust+Zero-Shot+Deep+Stereo+Matching+Even+Where+Either+Stereo+or+Mono+Fail&btnG=)]

   * **FoundationStereo**: *"FoundationStereo: Zero-Shot Stereo Matching"*, Wen et al., *CVPR, 2025*. [[Paper](https://arxiv.org/pdf/2501.09898)] [[Code](https://github.com/NVlabs/FoundationStereo)] [[WebPage](https://nvlabs.github.io/FoundationStereo/)] [[Bibtex](./bibliography/FoundationStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=FoundationStereo%3A+Zero-Shot+Stereo+Matching&btnG=#d=gs_cit&t=1737624791731&u=%2Fscholar%3Fq%3Dinfo%3AmPuArdM7kccJ%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Dit)]

   * **DEFOM-Stereo**: *"DEFOM-Stereo: Depth Foundation Model Based Stereo Matching"*, Jiang et al., *arXiv, 2025*. [[Paper](https://arxiv.org/pdf/2501.09466)] [[Code](https://github.com/Insta360-Research-Team/DEFOM-Stereo)] [[Bibtex](./bibliography/DEFOM-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=DEFOM-Stereo%3A+Depth+Foundation+Model+Based+Stereo+Matching&btnG=)]

   * **AIO-Stereo**: *"All-in-One: Transferring Vision Foundation Models into Stereo Matching"*, Zhou et al., *AAAI, 2025*. [[Paper](https://arxiv.org/pdf/2412.09912)] [[Bibtex](./bibliography/AIO-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=All-in-One%3A+Transferring+Vision+Foundation+Models+into+Stereo+Matching&btnG=)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Data Augmentation Analysis </summary>

* **NLCA-Net_v2**: "*Rethinking training strategy in stereo matching*", Rao et al., *TNNLS, 2022*. [[Paper](https://ieeexplore.ieee.org/iel7/5962385/10273172/09709604.pdf?casa_token=Bs47IDDe0r8AAAAA:J0ru7iqCpc1X1HGE3xsJ6u5I1gTAU8Z_wFAE_4QYg0-50X0dCUHvqrp7l8FZVsROvqZYwayc)] [[Code](https://github.com/Archaic-Atom/NLCA-Net_v2)] [[Bibtex](./bibliography/NLCA-Net_v2.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Rethinking+Training+Strategy+in+Stereo+Matching&btnG=)]


</details>

</ul>
</details>


<details open>
<summary style="font-size: larger; font-weight: bold;">Offline Adaptation</summary>

* **Confidence-guided Adaptation**: *"Unsupervised adaptation for deep stereo"*, Tonioni et al., *ICCV, 2017*. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tonioni_Unsupervised_Adaptation_for_ICCV_2017_paper.pdf)] [[Code](https://github.com/CVLAB-Unibo/Unsupervised-Adaptation-for-Deep-Stereo)] [[Bibtex1](./bibliography/Confidence_guided_Adaptation_0.txt)]  [[Bibtex2](./bibliography/Confidence_guided_Adaptation_1.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Unsupervised+adaptation+for+deep+stereo&btnG=)] 

* **Open-World Stereo**: *"Open-world stereo video matching with deep rnn"*, Zhong et al., *ECCV, 2018*. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yiran_Zhong_Open-World_Stereo_Video_ECCV_2018_paper.pdf)] [[Bibtex](./bibliography/Open-World.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Open-world+stereo+video+matching+with+deep+rnn&btnG=)] 

* **ZOLE**: *"Zoom and Learn: Generalizing Deep Stereo Matching to Novel Domain"*, Pang et al., *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Pang_Zoom_and_Learn_CVPR_2018_paper.pdf)] [[Code](https://github.com/jiahaopang/zole)] [[Bibtex](./bibliography/v.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Unsupervised+adaptation+for+deep+stereo&btnG=)]

* **StereoGAN**: "*StereoGAN: Bridging Synthetic-to-Real Domain Gap by Joint Optimization of Domain Translation and Stereo Matching*", Liu et al.,  *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_StereoGAN_Bridging_Synthetic-to-Real_Domain_Gap_by_Joint_Optimization_of_Domain_CVPR_2020_paper.pdf)] [[Code](https://github.com/ruiliu-ai/StereoGAN)] [[Bibtex](./bibliography/StereoGAN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=StereoGAN%3A+Bridging+Synthetic-to-Real+Domain+Gap+by+Joint+Optimization+of+Domain+Translation+and+Stereo+Matching&btnG=)] 

* **AdaStereo**: *"AdaStereo: A Simple and Efficient Approach for Adaptive Stereo Matching"*, Song et al., *CVPR, 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Song_AdaStereo_A_Simple_and_Efficient_Approach_for_Adaptive_Stereo_Matching_CVPR_2021_paper.pdf)] [[Bibtex](./bibliography/AdaStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=AdaStereo:+A+Simple+and+Efficient+Approach+for+Adaptive+Stereo+Matching&btnG=)]

* **UnDAF**: *"UnDAF: A General Unsupervised Domain Adaptation Framework for Disparity or Optical Flow Estimation"*, Wang et al., *CVPR, 2021*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9811811)] [[Code](https://sites.google.com/view/undaf)] [[Bibtex](./bibliography/UnDAF.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=UnDAF:+A+General+Unsupervised+Domain+Adaptation+Framework+for+Disparity+or+Optical+Flow+Estimation&btnG=)]

* **RAG**: *"Continual Stereo Matching of Continuous Driving Scenes With Growing Architecture"*, Zhang et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Continual_Stereo_Matching_of_Continuous_Driving_Scenes_With_Growing_Architecture_CVPR_2022_paper.pdf)] [[Bibtex](./bibliography/RAG.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Continual+Stereo+Matching+of+Continuous+Driving+Scenes+With+Growing+Architecture&btnG=)]

* **UCFNet**: *"Digging Into Uncertainty-Based Pseudo-Label for Robust Stereo Matching"*, Shen et al., *TPAMI, 2023*. [[Paper](https://arxiv.org/pdf/2307.16509.pdf)] [[Code](https://github.com/gallenszl/UCFNet?tab=readme-ov-file)] [[Bibtex](./bibliography/UCFNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Digging+Into+Uncertainty-Based+Pseudo-Label+for+Robust+Stereo+Matching&btnG=)]

* **StereoFlowGAN**: *"StereoFlowGAN: Co-training for Stereo and Flow with Unsupervised Domain Adaptation"*, Xiong et al., *BMVC, 2023*. [[Paper](https://papers.bmvc2023.org/0240.pdf)] [[Bibtex](./bibliography/StereoFlowGAN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=StereoFlowGAN:+Co-training+for+Stereo+and+Flow+with+Unsupervised+Domain+Adaptation&btnG=)]

* **Few-Shot Stereo Matching**: *"Few-Shot Stereo Matching with High Domain Adaptability Based on Adaptive Recursive Network"*, Wu et al.,*IJCV, 2023*. [[Paper](https://link.springer.com/content/pdf/10.1007/s11263-023-01953-0.pdf)]  [[Code](https://github.com/wu258/Few-shot-Stereo-Matching-with-High-Domain-Adaptability-Based-on-Adaptive-Recursive-Network)] [[Bibtex](./bibliography/Few-Shot.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Few-Shot+Stereo+Matching+with+High+Domain+Adaptability+Based+on+Adaptive+Recursive+Network&btnG=)]

* **RAG-Continual**: *"Reusable Architecture Growth for Continual Stereo Matching"*, Zhang et al.,*TPAMI, 2024*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10475553)]  [[Code](https://github.com/chzhang18/RAG)] [[Bibtex](./bibliography/RAG-Continual.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Reusable+Architecture+Growth+for+Continual+Stereo+Matching&btnG=)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Online Continual Adaptation</summary>


* :triangular_flag_on_post: **MadNet**: *"Real-Time Self-Adaptive Deep Stereo"*, Tonioni et al., *CVPR, 2019*. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tonioni_Real-Time_Self-Adaptive_Deep_Stereo_CVPR_2019_paper.pdf)] [[Code](https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo?tab=readme-ov-file)] [[Bibtex](./bibliography/MadNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Real-Time+Self-Adaptive+Deep+Stereo&btnG=)]

* **Learning2Adapt**: *"Learning to adapt for stereo"*, Tonioni et al., *CVPR, 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tonioni_Learning_to_Adapt_for_Stereo_CVPR_2019_paper.pdf)] [[Code](https://github.com/CVLAB-Unibo/Learning2AdaptForStereo)] [[Bibtex](./bibliography/Learning2Adapt.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+to+adapt+for+stereo&btnG=)]

* **AoHNet**: *"Faster Self-adaptive Deep Stereo"*, Wang et al., *ACCV, 2020*. [[Paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Wang_Faster_Self-adaptive_Deep_Stereo_ACCV_2020_paper.pdf)] [[Bibtex](./bibliography/AoHNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Faster+Self-adaptive+Deep+Stereo&btnG=)]


* **Continual Adaptation for Deep Stereo**: *"Continual adaptation for deep stereo"*, Poggi et al., *TPAMI, 2021*. [[Paper](https://ieeexplore.ieee.org/document/9418523?denied=)] [[Code](https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo?tab=readme-ov-file)] [[Bibtex](./bibliography/ContinualAdaptation.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Continual+adaptation+for+deep+stereo&btnG=)]


* **PointFix**: *"PointFix: Learning to Fix Domain Bias for Robust Online Stereo Adaptation"*, Kim et al., *ECCV, 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136980557.pdf)] [[Bibtex](./bibliography/PointFix.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=PointFix:+Learning+to+Fix+Domain+Bias+for+Robust+Online+Stereo+Adaptation&btnG=)]

* **FedStereo**: *"Federated Online Adaptation for Deep Stereo"*, Poggi et al., *CVPR, 2024*. [[Paper](https://fedstereo.github.io/)] [[Code](https://github.com/mattpoggi/fedstereo)] [[Bibtex](./bibliography/FedStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Federated+Online+Adaptation+for+Deep+Stereo&btnG=)]



</details>

</details>



<details open id="adverse-weather">
<summary style="font-size: larger; font-weight: bold;">Adverse Weather</summary>

* **FoggyStereo**: "*FoggyStereo: Stereo Matching with Fog Volume Representation*", Yao et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yao_FoggyStereo_Stereo_Matching_With_Fog_Volume_Representation_CVPR_2022_paper.pdf)] [[Code](https://github.com/nianticlabs/stereo-from-mono/)] [[Bibtex](./bibliography/FoggyStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=FoggyStereo%3A+Stereo+Matching+with+Fog+Volume+Representation&btnG=)] 

* **DDF**: *"Dusk Till Dawn: Self-supervised Nighttime Stereo Depth Estimation using Visual Foundation Models"*, Vankadari et al., *ICRA, 2024*. [[Paper](https://arxiv.org/pdf/2405.11158)]  [[Code](https://github.com/madhubabuv/dtd)] [[Bibtex](./bibliography/Dusk-Till-Dawn.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Dusk+Till+Dawn%3A+Self-supervised+Nighttime+Stereo+Depth+Estimation+using+Visual+Foundation+Models&btnG=)]


</details>






<details open id="tom">
<summary style="font-size: larger; font-weight: bold;">Transparent and Reflective (ToM) Surfaces </summary>

* **DDF**: *"Deep Depth Fusion for Black, Transparent, Reflective and Texture-Less Objects"*, Chai et al., *ICRA, 2020*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9196894)] [[Bibtex](./bibliography/DDF.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Deep+Depth+Fusion+for+Black,+Transparent,+Reflective+and+Texture-Less+Objects&btnG=)]

* **TA-Stereo**: *"Transparent Objects: A Corner Case in Stereo Matching"*, Wu et al., *ICRA, 2023*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10161385)]  [[Code](mias.group/TA-Stereo)] [[Bibtex](./bibliography/TA-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Transparent+Objects:+A+Corner+Case+in+Stereo+Matching&btnG=)]

* **Depth4ToM**: *"Learning Depth Estimation for Transparent and Mirror Surfaces"*, Costanzino et al., *ICCV, 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Costanzino_Learning_Depth_Estimation_for_Transparent_and_Mirror_Surfaces_ICCV_2023_paper.pdf)]  [[Code](https://cvlab-unibo.github.io/Depth4ToM/)] [[Bibtex](./bibliography/Depth4ToM.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+Depth+Estimation+for+Transparent+and+Mirror+Surfaces&btnG=)]

* **ASGrasp**: *"ASGrasp: Generalizable Transparent Object Reconstruction and 6-DoF Grasp Detection from RGB-D Active Stereo Camera"*, Shi et al., *ICRA, 2024*. [[Paper](https://arxiv.org/pdf/2405.05648)] [[WebPage](https://pku-epic.github.io/ASGrasp)] [[Bibtex](./bibliography/ASGrasp.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=ASGrasp%3A+Generalizable+Transparent+Object+Reconstruction+and+Grasping+from+RGB-D+Active+Stereo+Camera&btnG=)]

* **D3RoMa**: *"D3RoMa: Disparity Diffusion-based Depth Sensing for Material-Agnostic Robotic Manipulation"*, Wei et al., *CoRL, 2024*. [[Paper](https://arxiv.org/pdf/2405.05648)] [[WebPage](https://pku-epic.github.io/D3RoMa/)]  [[Bibtex](./bibliography/D3RoMa.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=D3RoMa%3A+Disparity+Diffusion-based+Depth+Sensing+for+Material-Agnostic+Robotic+Manipulation&btnG=)]

* **Stereo Anywhere**: *"Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching Even Where Either Stereo or Mono Fail"*, Bartolomei et al., *CVPR, 2025*. [[Paper](http://arxiv.org/abs/2412.04472)] [[Code](https://github.com/bartn8/stereoanywhere/)] [[WebPage](https://stereoanywhere.github.io/)] [[Bibtex](./bibliography/StereoAnywhere.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Stereo+Anywhere%3A+Robust+Zero-Shot+Deep+Stereo+Matching+Even+Where+Either+Stereo+or+Mono+Fail&btnG=)]

</details>


<details open id="asymmetric">
<summary style="font-size: larger; font-weight: bold;">Asymmetric Stereo </summary>

* **Visually-Imbalanced Stereo**: "*Visually Imbalanced Stereo Matching*", Liu et al., *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Visually_Imbalanced_Stereo_Matching_CVPR_2020_paper.pdf)] [[Code](https://github.com/DandilionLau/Visually-Imbalanced-Stereo)] [[Bibtex](./bibliography/VI-SM.txt)] 

* **NDR**: "*Neural disparity refinement for arbitrary resolution stereo*", Aleotti et al., *3DV, 2021*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9665913?casa_token=3rm4WpqLb_QAAAAA:5Sa0RO547j8LsaEYUeppzB33gZJg5Y3tfiPVwM9rzs9MEAuoHSta0Kdw3Cm9NrtfOOdFkIwp)] [[Website](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Neural+disparity+refinement+for+arbitrary+resolution+stereo&btnG=)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Neural+disparity+refinement+for+arbitrary+resolution+stereo&btnG=)]  

* **DA-AS**: *"Degradation-agnostic Correspondence from Resolution-asymmetric Stereo"*, Chen et al., *CVPR, 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Degradation-Agnostic_Correspondence_From_Resolution-Asymmetric_Stereo_CVPR_2022_paper.pdf)] [[Bibtex](./bibliography/DA-AS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Degradation-agnostic+Correspondence+from+Resolution-asymmetric+Stereo&btnG=)]

* **SASS**: *"Unsupervised Deep Asymmetric Stereo Matching with Spatially-Adaptive Self-Similarity"*, Song et al., *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Song_Unsupervised_Deep_Asymmetric_Stereo_Matching_With_Spatially-Adaptive_Self-Similarity_CVPR_2023_paper.pdf)] [[Bibtex](./bibliography/SASS.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Unsupervised+Deep+Asymmetric+Stereo+Matching+with+Spatially-Adaptive+Self-Similarity&btnG=)]


* **NDR v2**: "*Neural disparity refinement*", Tosi et al., *TPAMI, 2024*. [[Paper](https://ieeexplore.ieee.org/iel8/34/4359286/10552115.pdf)] [[Website](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR_v2.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Neural+disparity+refinement&btnG=)]  

</details>

<details open id="temporal-consistency">
<summary style="font-size: larger; font-weight: bold;">Temporal Consistency</summary>


   * **TemporalStereo**: *"TemporalStereo: Efficient Spatial-Temporal Stereo Matching Network"*, Zhang et al., *IROS, 2023*. [[Paper](https://arxiv.org/pdf/2211.13755.pdf)] [[Code](https://github.com/youmi-zym/TemporalStereo)]  [[Bibtex](./bibliography/TemporalStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=TemporalStereo:+Efficient+Spatial-Temporal+Stereo+Matching+Network&btnG=)]

   * **Dynamic Stereo**: *"DynamicStereo: Consistent Dynamic Depth From Stereo Videos"*, Karaev et al., *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Karaev_DynamicStereo_Consistent_Dynamic_Depth_From_Stereo_Videos_CVPR_2023_paper.pdf)] [[Code](https://dynamic-stereo.github.io/)] [[Bibtex](./bibliography/Dynamic_Replica.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=DynamicStereo:+Consistent+Dynamic+Depth+From+Stereo+Videos&btnG=)]

   * **XR-Stereo**: *"Stereo Matching in Time: 100+ FPS Video Stereo Matching for Extended Reality"*, Cheng et al., *WACV, 2024*. [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Cheng_Stereo_Matching_in_Time_100_FPS_Video_Stereo_Matching_for_WACV_2024_paper.pdf)] [[Code](https://github.com/za-cheng/XR-Stereo)] [[Bibtex](./bibliography/XR-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Stereo+Matching+in+Time:+100++FPS+Video+Stereo+Matching+for+Extended+Reality&btnG=)]

   * **TemporalEventStereo**: *"Temporal Event Stereo via Joint Learning with Stereoscopic Flow"*, Cho et al., *ECCV, 2024*. [[Paper](https://arxiv.org/pdf/2407.10831)] [[Code](https://github.com/mickeykang16/TemporalEventStereo)] [[Bibtex](./bibliography/TemporalEventStereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Temporal+Event+Stereo+via+Joint+Learning+with+Stereoscopic+Flow&btnG=)]

   * **Temporally-Consistent Stereo**: *"Temporally Consistent Stereo Matching"*, Zeng et al., *ECCV, 2024*. [[Paper](https://arxiv.org/pdf/2407.11950)] [[Code](https://github.com/jiaxiZeng/Temporally-Consistent-Stereo-Matching)] [[Bibtex](./bibliography/Temporally-Consistent-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Temporally+Consistent+Stereo+Matching&btnG=)]

   * **BiDA-Stereo**: *"Match-Stereo-Videos: Bidirectional Alignment for Consistent Dynamic Stereo Matching"*, Jing et al., *ECCV, 2024*. [[Paper](https://arxiv.org/abs/2403.10755)] [[Code](https://github.com/TomTomTommi/bidastereo)] [[Bibtex](./bibliography/Match-Stereo-Videos.txt)] [[WebPage](https://tomtomtommi.github.io/BiDAStereo/)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Match-Stereo-Videos%3A+Bidirectional+Alignment+for+Consistent+Dynamic+Stereo+Matching&btnG=)]

   * **BiDA-Video**: *"Match Stereo Videos via Bidirectional Alignment"*, Jing et al., *arXiv, 2024*. [[Paper](https://arxiv.org/pdf/2409.20283)] [[Code](https://github.com/TomTomTommi/bidavideo)] [[Bibtex](./bibliography/BiDA-Video.txt)] [[WebPage](https://tomtomtommi.github.io/BiDAVideo/)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Match+Stereo+Videos+via+Bidirectional+Alignment&btnG=)]

   * **Stereo Any Video**: *"Stereo Any Video: Temporally Consistent Stereo Matching"*, Jing et al., *arXiv, 2025*. [[Paper](https://arxiv.org/pdf/2503.05549)] [[Code](https://github.com/TomTomTommi/stereoanyvideo)] [[Bibtex](./bibliography/StereoAnyVideo.txt)] [[WebPage](https://tomtomtommi.github.io/StereoAnyVideo/)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Stereo+Any+Video%3A+Temporally+Consistent+Stereo+Matching&btnG=)]

</details>

<details open id="continuous-estimation">
<summary style="font-size: larger; font-weight: bold;">Continuous Estimation Problem</summary>

* **SMD-Nets**: *"SMD-Nets: Stereo Mixture Density Networks"*, Tosi et al.,  *CVPR, 2021*. [[Paper](http://www.cvlibs.net/publications/Tosi2021CVPR.pdf)] [[Code](https://github.com/fabiotosi92/SMD-Nets)] [[Bibtex](./bibliography/SMD-Nets.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=SMD-Nets:+Stereo+Mixture+Density+Networks&btnG=)] 

* **NDR**: "*Neural disparity refinement for arbitrary resolution stereo*", Aleotti et al., *3DV, 2021*. [[Paper](https://ieeexplore.ieee.org/abstract/document/9665913?casa_token=3rm4WpqLb_QAAAAA:5Sa0RO547j8LsaEYUeppzB33gZJg5Y3tfiPVwM9rzs9MEAuoHSta0Kdw3Cm9NrtfOOdFkIwp)] [[Website](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Neural+disparity+refinement+for+arbitrary+resolution+stereo&btnG=)]  


* **NDR v2**: "*Neural disparity refinement*", Tosi et al., *TPAMI, 2024*. [[Paper](https://ieeexplore.ieee.org/iel8/34/4359286/10552115.pdf)] [[Website](https://cvlab-unibo.github.io/neural-disparity-refinement-web/)] [[Bibtex](./bibliography/NDR_v2.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Neural+disparity+refinement&btnG=)]  


</details>

</ul>
</details>




### Confidence Estimation

<details open>
<summary style="font-size: larger; font-weight: bold;">Machine Learning Approaches</summary><ul>

<details open>
<summary style="font-size: larger; font-weight: bold;">Disparity-based</summary>

* **ENS7**: *"Ensemble learning for confidence measures in stereo vision"*, Haeusler et al., CVPR, 2013. [[Paper](http://openaccess.thecvf.com/content_cvpr_2013/papers/Haeusler_Ensemble_Learning_for_2013_CVPR_paper.pdf)]  [[Bibtex](./bibliography/ENS23.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Ensemble+learning+for+confidence+measures+in+stereo+vision&btnG=)]

* **O1**: *"Learning a general-purpose confidence measure based on o (1) features and a smarter aggregation strategy for semi global matching"*, Poggi et al., 3DV, 2016. [[Paper](http://vision.disi.unibo.it/~mpoggi/papers/3dv2016_o1.pdf)]  [[Bibtex1](./bibliography/O1.txt)] [[Bibtex2](./bibliography/O1_1.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Learning+a+general-purpose+confidence+measure+based+on+o+%281%29+features+and+a+smarter+aggregation+strategy+for+semi+global+matching&btnG=)]


</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Cost Volume-based</summary>

* **ENS23**: *"Ensemble learning for confidence measures in stereo vision"*, Haeusler et al., CVPR, 2013. [[Paper](http://openaccess.thecvf.com/content_cvpr_2013/papers/Haeusler_Ensemble_Learning_for_2013_CVPR_paper.pdf)]  [[Bibtex](./bibliography/ENS23.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+a+general-purpose+confidence+measure+based+on+o+(1)+features+and+a+smarter+aggregation+strategy+for+semi+global+matching&btnG=)]

* **GCP**: *"Learning to detect ground control points for improving the accuracy of stereo matching"*, Spyropoulos et al., CVPR, 2014. [[Paper](https://openaccess.thecvf.com/content_cvpr_2014/papers/Spyropoulos_Learning_to_Detect_2014_CVPR_paper.pdf)]  [[Bibtex1](./bibliography/GCP.txt)] [[Bibtex2](./bibliography/GCP1.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Learning+to+detect+ground+control+points+for+improving+the+accuracy+of+stereo+matching&btnG=)]

* **LEV**: *"Leveraging stereo matching with learning-based confidence measures"*, Park et al., CVPR, 2015. [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Park_Leveraging_Stereo_Matching_2015_CVPR_paper.pdf)]  [[Bibtex1](./bibliography/LEV.txt)] [[Bibtex2](./bibliography/LEV1.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Leveraging+stereo+matching+with+learning-based+confidence+measures&btnG=)]

* **FA**: *"Feature augmentation for learning confidence measure in stereo matching"*, Kim et al., TIP, 2017. [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Park_Leveraging_Stereo_Matching_2015_CVPR_paper.pdf)]  [[Bibtex](./bibliography/FA.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+to+detect+ground+control+points+for+improving+the+accuracy+of+stereo+matching&btnG=)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Model-based</summary>

* **Multi-Task Learning Using Uncertainty**: *"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"*, Kendall et al., CVPR, 2018. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)]  [[Bibtex](./bibliography/Multi-Task-Learning.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=2005&sciodt=0%2C5&cites=502631552324336717&scipsc=&q=Multi-Task+Learning+Using+Uncertainty+to+Weigh+Losses+for+Scene+Geometry+and+Semantics&btnG=)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">SGM-specific</summary>

* **SGMForest**: *"Learning to fuse proposals from multiple scanline optimizations in semi-global matching"*, Schonberger et al., ECCV, 2018. [[Paper](http://vision.disi.unibo.it/~mpoggi/papers/3dv2016_o1.pdf)]  [[Bibtex](./bibliography/SGMForest.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+to+fuse+proposals+from+multiple+scanline+optimizations+in+semi-global+matching&btnG=)]


</details>

</ul>
</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Deep Learning Approaches</summary><ul>

<details open>
<summary style="font-size: larger; font-weight: bold;">Disparity-based</summary>

* **CCNN**: *"Learning from scratch a confidence measure"*, Poggi et al., BMVC, 2016. [[Paper](https://www.researchgate.net/profile/Stefano-Mattoccia-2/publication/317191595_Learning_from_scratch_a_confidence_measure/links/593f9db3a6fdcc1b10aac9ec/Learning-from-scratch-a-confidence-measure.pdf)] [[Code](https://github.com/fabiotosi92/CCNN-Tensorflow)] [[Bibtex](./bibliography/CCNN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+from+scratch+a+confidence+measure&btnG=)]

* **PBCP**: *"Patch Based Confidence Prediction for Dense Disparity Map"*, Seki et al., BMVC, 2016. [[Paper](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Seki2016BMVC.pdf)]  [[Bibtex](./bibliography/PBCP.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Patch+Based+Confidence+Prediction+for+Dense+Disparity+Map&btnG=)]

* **EFN/LFN**: *"Stereo matching confidence learning based on multi-modal convolution neural networks"*, Fu et al., RFMI, 2017. [[Paper](http://www.arts-pi.org.tn/rfmi2017/papers/10_CameraReadySubmission_llncs2e%20(3).pdf)] [[Bibtex](./bibliography/EFN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Stereo+matching+confidence+learning+based+on+multi-modal+convolution+neural+networks&btnG=)]

* **MMC**: *"Learning confidence measures by multi-modal convolutional neural networks"*, Fu et al., WACV, 2018. [[Paper](WACV)] [[Bibtex](./bibliography/MMC.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+confidence+measures+by+multi-modal+convolutional+neural+networks&btnG=)]

* **LGC/ConfNet**: *"Beyond local reasoning for stereo confidence estimation with deep learning"*, Tosi et al.,  ECCV, 2018. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Fabio_Tosi_Beyond_local_reasoning_ECCV_2018_paper.pdf)] [[Code](https://github.com/fabiotosi92/LGC-Tensorflow)] [[Bibtex](./bibliography/LGC.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Beyond+local+reasoning+for+stereo+confidence+estimation+with+deep+learning&btnG=)]

* **Self-adapting Confidence**: *"Self-adapting confidence estimation for stereo"*, Poggi et al.,  ECCV, 2020. [[Paper](https://arxiv.org/pdf/2008.06447.pdf)] [[Code](https://github.com/mattpoggi/self-adapting-confidence)] [[Bibtex](./bibliography/Self-adapting-confidence.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Self-adapting+confidence+estimation+for+stereo&btnG=)]

* **SEDNet**: *"Learning the distribution of errors in stereo matching for joint disparity and uncertainty estimation"*, Chen et al., CVPR, 2023. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Learning_the_Distribution_of_Errors_in_Stereo_Matching_for_Joint_CVPR_2023_paper.pdf)] [[Code](https://github.com/lly00412/SEDNet)] [[Bibtex](./bibliography/SEDNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+the+distribution+of+errors+in+stereo+matching+for+joint+disparity+and+uncertainty+estimation&btnG=)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Cost Volume-based</summary>

* **RCN**: *"Improved stereo matching with constant highway networks and reflective confidence learning"*, Shaked et al., CVPR, 2017. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Shaked_Improved_Stereo_Matching_CVPR_2017_paper.pdf)] [[Code](https://github.com/amitshaked/resmatch)] [[Bibtex](./bibliography/RCN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Improved+stereo+matching+with+constant+highway+networks+and+reflective+confidence+learning&btnG=)]

* **MPN**: *"Deep stereo confidence prediction for depth estimation"*, Kim et al.,  ICIP, 2017. [[Paper](https://ieeexplore.ieee.org/iel7/8267582/8296222/08296430.pdf)] [[Bibtex](./bibliography/MPN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Deep+stereo+confidence+prediction+for+depth+estimation&btnG=)]

* **UCN**: *"Unified confidence estimation networks for robust stereo matching"*, Kim et al., TIP, 2018. [[Paper](https://ieeexplore.ieee.org/iel7/83/4358840/08510870.pdf)] [[Bibtex](./bibliography/UCN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Unified+confidence+estimation+networks+for+robust+stereo+matching&btnG=)]

* **LAF**: *"Laf-net: Locally adaptive fusion networks for stereo confidence estimation"*, Kim et al.,  CVPR, 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_LAF-Net_Locally_Adaptive_Fusion_Networks_for_Stereo_Confidence_Estimation_CVPR_2019_paper.pdf)] [[Bibtex](./bibliography/LAF.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Laf-net:+Locally+adaptive+fusion+networks+for+stereo+confidence+estimation&btnG=)]

* **CRNN**: *"Pixel-Wise Confidences for Stereo Disparities Using Recurrent Neural Networks"*, Gul et al.,  BMVC, 2019. [[Paper](https://publica.fraunhofer.de/bitstreams/c1f200e0-49e6-488c-84de-d217a550bdf6/download)] [[Bibtex](./bibliography/CRNN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Pixel-Wise+Confidences+for+Stereo+Disparities+Using+Recurrent+Neural+Networks&btnG=)]

* **CVA**: *"Cnn-based cost volume analysis as confidence measure for dense matching"*, Mehltretter et al., ICCVW, 2019. [[Paper](http://openaccess.thecvf.com/content_ICCVW_2019/papers/3DRW/Mehltretter_CNN-Based_Cost_Volume_Analysis_as_Confidence_Measure_for_Dense_Matching_ICCVW_2019_paper.pdf)] [[Bibtex](./bibliography/CVA.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Cnn-based+cost+volume+analysis+as+confidence+measure+for+dense+matching&btnG=)]

* **Disparity Plane Sweep**: *"Modeling Stereo-Confidence Out of the End-to-End Stereo-Matching Network via Disparity Plane Sweep"*, Lee et al., AAAI, 2024. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28071/28148)] [[Bibtex](./bibliography/Disparity_Plane_Sweep.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Modeling+Stereo-Confidence+Out+of+the+End-to-End+Stereo-Matching+Network+via+Disparity+Plane+Sweep&btnG=)]

* **ACN**: *"Adversarial confidence estimation networks for robust stereo matching"*, Kim et al., T-ITS, 2020. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_LAF-Net_Locally_Adaptive_Fusion_Networks_for_Stereo_Confidence_Estimation_CVPR_2019_paper.pdf)] [[Bibtex](./bibliography/ACN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Adversarial+confidence+estimation+networks+for+robust+stereo+matching&btnG=)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Multiple Confidence Fusion</summary>

* **Learning Local Consistency**: *"Learning to predict stereo reliability enforcing local consistency of confidence maps"*, Poggi et al., CVPR, 2017. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Poggi_Learning_to_Predict_CVPR_2017_paper.pdf)] [[Bibtex](./bibliography/++.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Learning+to+predict+stereo+reliability+enforcing+local+consistency+of+confidence+maps&btnG=)]

* **EMC**: *"Even More Confident Predictions With Deep Machine-Learning"*, Poggi et al., CVPRW, 2017. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Poggi_Learning_to_Predict_CVPR_2017_paper.pdf)] [[Bibtex](./bibliography/EMC.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Even+More+Confident+Predictions+With+Deep+Machine-Learning&btnG=)]

</details>

<details open>
<summary style="font-size: larger; font-weight: bold;">Sensor-based</summary>


* **Lidar-Confidence**: *"Unsupervised confidence for lidar depth maps and applications"*, Conti et al., IROS, 2022. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9981654)] [[Bibtex](./bibliography/Lidar-Confidence.txt)] [[Code](https://github.com/andreaconti/lidar-confidence)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Unsupervised+confidence+for+LiDAR+depth+maps+and+applications&btnG=)]


</ul>
</details>


## Applications

(Not an exhaustive list)

* **Deep3d**: *"Deep3d: Fully automatic 2d-to-3d video conversion with deep convolutional neural networks"*, Xie et al., *ECCV, 2016*. [[Paper](https://arxiv.org/pdf/1604.03650.pdf)] [[Code](https://github.com/piiswrong/deep3d)] [[Bibtex](./bibliography/Deep3d.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=eep3d%3A+Fully+automatic+2d-to-3d+video+conversion+with+deep+convolutional+neural+network&btnG=)]

* **Geometry to the Rescue**: *"Unsupervised cnn for single view depth estimation: Geometry to the rescue"*, Garg et al., *ECCV, 2016*. [[Paper](https://arxiv.org/pdf/1603.04992)] [[Bibtex](./bibliography/Geometry_to_the_Rescue.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Unsupervised+CNN+for+single+view+depth+estimation%3A+Geometry+to+the+rescu&btnG=)]

* **MonoDepth/StereoDepth**: *"Unsupervised monocular depth estimation with left-right consistency"*, Godard et al., *CVPR, 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Godard_Unsupervised_Monocular_Depth_CVPR_2017_paper.pdf)] [[Code](https://github.com/mrharicot/monodepth)] [[Bibtex](./bibliography/MonoDepth.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Unsupervised+monocular+depth+estimation+with+left-right+consistency&btnG=)]

* **SVSM**: "Single View Stereo Matching", Luo et al., *CVPR, 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Single_View_Stereo_CVPR_2018_paper.pdf)] [[Code](https://github.com/yanqi1811/Single-View-Stereo-Matching)] [[Bibtex](./bibliography/SVSM.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Single+View+Stereo+Matching&btnG=)]

* **MonoResMatch**: "Learning monocular depth estimation infusing traditional stereo knowledge", Tosi et al., *CVPR, 2019*. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tosi_Learning_Monocular_Depth_Estimation_Infusing_Traditional_Stereo_Knowledge_CVPR_2019_paper.pdf)] [[Code](https://github.com/fabiotosi92/monoResMatch-Tensorflow)] [[Bibtex](./bibliography/MonoResMatch.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Learning+monocular+depth+estimation+infusing+traditional+stereo+knowledge&btnG=)]

* **Ida-3d**: "Ida-3d: Instance-depth-aware 3d object detection from stereo vision for autonomous driving", Peng et al., *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Peng_IDA-3D_Instance-Depth-Aware_3D_Object_Detection_From_Stereo_Vision_for_Autonomous_CVPR_2020_paper.pdf)] [[Code](https://github.com/swords123/IDA-3D)] [[Bibtex](./bibliography/Ida-3d.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=IDA-3D%3A+Instance-Depth-Aware+3D+Object+Detection+From+Stereo+Vision+for+Autonomous+Driving&btnG=)]

* **LIGA-Stereo**: "LIGA-Stereo: Learning Lidar Geometry aware Representations for Stereo-based 3d Detector", Guo et al., *ICCV, 2021*. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_LIGA-Stereo_Learning_LiDAR_Geometry_Aware_Representations_for_Stereo-Based_3D_Detector_ICCV_2021_paper.pdf)] [[Code](https://github.com/xy-guo/LIGA-Stereo)] [[Bibtex](./bibliography/LIGA-Stereo.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=LIGA-Stereo%3A+Learning+LiDAR+Geometry+Aware+Representations+for+Stereo-based+3D+Detector&btnG=)]

* **Stereopifu**: "Stereopifu: Depth aware clothed human digitization via stereo vision", Hong et al., *CVPR, 2021*. [[Paper](http://openaccess.thecvf.com/content/CVPR2021/papers/Hong_StereoPIFu_Depth_Aware_Clothed_Human_Digitization_via_Stereo_Vision_CVPR_2021_paper.pdf)] [[Code](https://github.com/CrisHY1995/StereoPIFu_Code)] [[Bibtex](./bibliography/Stereopifu.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=StereoPIFu%3A+Depth+Aware+Clothed+Human+Digitization+via+Stereo+Vision&btnG=)]

* **Smart Glasses**: "A Practical Stereo Depth System for Smart Glasses", Wang et al., *CVPR, 2023*. [[Paper](http://openaccess.thecvf.com/content/CVPR2023/papers/Wang_A_Practical_Stereo_Depth_System_for_Smart_Glasses_CVPR_2023_paper.pdf)] [[Bibtex](./bibliography/Smart_Glasses.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=A+Practical+Stereo+Depth+System+for+Smart+Glasses&btnG=)]

* **Cross Attention Renderer**: "Learning to render novel views from wide-baseline stereo pairs", Du et al., *CVPR, 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Du_Learning_To_Render_Novel_Views_From_Wide-Baseline_Stereo_Pairs_CVPR_2023_paper.pdf)] [[Code](https://github.com/yilundu/cross_attention_renderer)] [[Bibtex](./bibliography/Cross_Attention_Renderer.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Learning+To+Render+Novel+Views+From+Wide-Baseline+Stereo+Pairs&btnG=)]

* **SDCNet**: "Stereo-augmented depth completion from a single rgb-lidar image", Choi et al., *ICRA, 2021*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9561557)] [[Bibtex](./bibliography/SDCNet.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Stereo-augmented+depth+completion+from+a+single+rgb-lidar+image&btnG=)]

* **VPPDC**: "Revisiting Depth Completion from a Stereo Matching Perspective for Cross-domain Generalization", Bartolomei et al., *3DV, 2024*. [[Paper](https://vppdc.github.io/assets/paper.pdf)] [[Code](https://vppdc.github.io/)] [[Bibtex](./bibliography/VPPDC.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Revisiting+Depth+Completion+from+a+Stereo+Matching+Perspective+for+Cross-domain+Generalization&btnG=)]

* **CoPoNeRF**: "Unifying Correspondence, Pose and NeRF for Pose-Free Novel View Synthesis from Stereo Pairs", Hong et al., *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2312.07246v1)] [[Code](https://github.com/KU-CVLAB/CoPoNeRF)] [[Bibtex](./bibliography/CoPoNeRF.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Unifying+Correspondence%2C+Pose+and+NeRF+for+Pose-Free+Novel+View+Synthesis+from+Stereo+Pairs&btnG=)]

* **DSGN**: "Deep Stereo Geometry Network for 3D Object Detection", Chen et al., *CVPR, 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_DSGN_Deep_Stereo_Geometry_Network_for_3D_Object_Detection_CVPR_2020_paper.pdf)] [[Code](https://github.com/dvlab-research/DSGN)] [[Bibtex](./bibliography/DSGN.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Deep+Stereo+Geometry+Network+for+3D+Object+Detection&btnG=)]

* **StereoNeRF**: "Generalizable Novel-View Synthesis using a Stereo Camera", Lee et al., *CVPR, 2024*. [[Paper](https://arxiv.org/pdf/2404.13541.pdf)] [[WebSite](https://jinwonjoon.github.io/stereonerf/)] [[Bibtex](./bibliography/StereoNeRF.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Generalizable+Novel-View+Synthesis+using+a+Stereo+Camera&btnG=)]

* **Online Stereo Rectification**: "Flow-Guided Online Stereo Rectification for Wide Baseline Stereo", Kumar et al., *CVPR, 2024*. [[Paper](https://light.princeton.edu/wp-content/uploads/2024/04/Flow-Guided-Online-Stereo-Rectification-for-Wide-Baseline-Stereo.pdf)] [[WebSite](https://light.princeton.edu/publication/online-stereo-recification/)] [[Bibtex](./bibliography/OnlineStereoRectification.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Flow-Guided+Online+Stereo+Rectification+for+Wide+Baseline+Stereo&btnG=)]

* **GS2Mesh**: "*GS2Mesh: Surface Reconstruction from Gaussian
Splatting via Novel Stereo Views*", Wolf et al., *ECCV, 2024*. [[Paper](https://arxiv.org/pdf/2404.01810)] [[WebPage](https://gs2mesh.github.io/)] [[Bibtex](./bibliography/GS2Mesh.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=GS2Mesh%3A+Surface+Reconstruction+from+Gaussian+Splatting+via+Novel+Stereo+Views&btnG=)] 

* **StereoGS**: "Self-Evolving Depth-Supervised 3D Gaussian Splatting from Rendered Stereo Pairs", Safadoust et al., *BMVC, 2024*. [[Paper](https://arxiv.org/pdf/2409.07456)] [[WebPage](https://kuis-ai.github.io/StereoGS/)] [[Bibtex](./bibliography/StereoGS.txt)]  [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=Self-Evolving+Depth-Supervised+3D+Gaussian+Splatting+from+Rendered+Stereo+Pairs&btnG=)] 

* **Binocular3DGS**: "*Binocular3DGS: Binocular-Guided 3D Gaussian Splatting with View Consistency for Sparse View Synthesis*", Han et al., *NeurIPS, 2024*. [[Paper](https://arxiv.org/pdf/2410.18822)] [[WebPage](https://hanl2010.github.io/Binocular3DGS/)] [[Code](https://github.com/hanl2010/Binocular3DGS)] [[Bibtex](./bibliography/Binocular3DGS.txt)]  [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0,5&q=Binocular+3DGS:+Binocular-Guided+3D+Gaussian+Splatting+with+View+Consistency+for+Sparse+View+Synthesis)]

* **StereoDiffusion**: "StereoDiffusion: Training-Free Stereo Image Generation Using Latent Diffusion Models", Wang et al., *CVPRW, 2024*. [[Paper](https://openaccess.thecvf.com/content/CVPR2024W/GCV/papers/Wang_StereoDiffusion_Training-Free_Stereo_Image_Generation_Using_Latent_Diffusion_Models_CVPRW_2024_paper.pdf)] [[Code](https://github.com/lez-s/StereoDiffusion)] [[Bibtex](./bibliography/StereoDiffusion.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=StereoDiffusion%3A+Training-Free+Stereo+Image+Generation+Using+Latent+Diffusion+Models&btnG=)] 

* **StereoGen**: "StereoGen: High-quality Stereo Image Generation from a Single Image", Wang et al., *arXiv, 2025*. [[Paper](https://arxiv.org/pdf/2501.08654)] [[Bibtex](./bibliography/StereoGen.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=StereoGen%3A+High-quality+Stereo+Image+Generation+from+a+Single+Image&btnG=)] 

* **StereoCrafter**: "StereoCrafter: Diffusion-based Generation of Long and High-fidelity Stereoscopic 3D from Monocular Videos", Zhao et al., *arXiv, 2024*. [[Paper](https://stereocrafter.github.io/static/pdfs/StereoCrafter_report.pdf)] [[WebPage](https://stereocrafter.github.io/)] [[GitHub](https://github.com/TencentARC/StereoCrafter)] [[Bibtex](./bibliography/StereoCrafter.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=StereoCrafter%3A+Diffusion-based+Generation+of+Long+and+High-fidelity+Stereoscopic+3D+from+Monocular+Videos&btnG=)] 

* **StereoCrafter-Zero**: "StereoCrafter-Zero: Zero-Shot Stereo Video Generation with Noisy Restart", Shi et al., *arXiv, 2024*. [[Paper](https://arxiv.org/pdf/2411.14295?)] [[GitHub](https://github.com/shijianjian/StereoCrafter-Zero)] [[Bibtex](./bibliography/StereoCrafter-Zero.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=StereoCrafter-Zero%3A+Zero-Shot+Stereo+Video+Generation+with+Noisy+Restart&btnG=)] 

* **SpatialMe**: "SpatialMe: Stereo Video Conversion Using Depth-Warping and Blend-Inpainting", Zhang et al., *arXiv, 2024*. [[Paper](https://arxiv.org/pdf/2412.11512)] [[Bibtex](./bibliography/SpatialMe.txt)] [[Google Scholar](https://scholar.google.com/scholar?hl=it&as_sdt=0%2C5&q=SpatialMe%3A+Stereo+Video+Conversion+Using+Depth-Warping+and+Blend-Inpainting&btnG=)] 

## Workshops

* **NTIRE 2024: HR Depth from Images of Specular and Transparent Surfaces**. P. Z. Ramirez, F. Tosi, L. Di Stefano, R. Timofte
A. Costanzino, M. Poggi, S. Salti, S. Mattoccia; CVPRW 2024, Seattle, US [[Website](https://cvlab-unibo.github.io/booster-web/ntire.html)]

* **NTIRE 2023: HR Depth from Images of Specular and Transparent Surfaces**. P. Z. Ramirez, F. Tosi, L. Di Stefano, R. Timofte
A. Costanzino, M. Poggi, S. Salti, S. Mattoccia; CVPRW 2023, Vancouver, Canada [[Website](https://cvlab-unibo.github.io/booster-web/ntire.html)]

* **Robust Vision Challenge (ROB)**, Zendel et al., ECCV 2022 [[Website](http://www.robustvision.net/)]




<h2 id="tutorials-talks"> Tutorials & Talks </h2>

* **Deep Stereo Matching in the Twenties**. M. Poggi, F. Tosi; CVPR 2024, Seattle, US [[Website](https://sites.google.com/view/stereo-twenties/home)]

* **Facing depth estimation in-the-wild with deep networks**. M. Poggi, F. Tosi, F. Aleotti, K. Batsos, P. Mordohai, S. Mattoccia; ECCV 2020, SEC, Glasgow [[Website](https://sites.google.com/view/eccv-2020-robust-depth/home)]

* **Learning and understanding single image depth estimation in the wild**.  M. Poggi, F. Tosi, F. Aleotti, S. Mattoccia, C. Godard, J. Watson, M. Firman, G.J. Brostow; CVPR 2020, Seattle, Washington, US  [[Website](https://sites.google.com/view/cvpr-2020-depth-from-mono/home)]

* **Learning-based depth estimation from stereo and monocular images: successes, limitations and future challenges**. M. Poggi, F. Tosi, K. Batsos, P. Mordohai, S. Mattoccia, CVPR 2019, Long Beach, California, US [[Website](https://sites.google.com/view/cvpr-2019-depth-from-image/home)]

* **Learning-based depth estimation from stereo and monocular images: successes, limitations and future challenges**.  M. Poggi, F. Tosi, K. Batsos, P. Mordohai, S. Mattoccia; 3DV 2018, Verona, Italy [[Website](https://sites.google.com/view/3dv-2018-depth-from-image/home)]

* **Lecture: Computer Vision (Prof. Andreas Geiger, University of Tübingen)**. [[Preliminaries](https://www.youtube.com/watch?v=6hr6xpOU-uw&list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_&index=12&pp=iAQB)] [[Block Matching](https://www.youtube.com/watch?v=EVzEJQl8WFk&list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_&index=13&t=830s&pp=iAQB)] [[Siamese Networks](https://www.youtube.com/watch?v=vLgsiIXNf0I&list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_&index=14&pp=iAQB)] [[Spatial Regularization](https://www.youtube.com/watch?v=gqz6R1qChVQ&list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_&index=15&t=359s&pp=iAQB)] [[End-to-End Learning](https://www.youtube.com/watch?v=9vrmwZ9Pl4o&list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_&index=16&t=639s&pp=iAQB)] 



<h2 id="demos"> Demos </h2>

* **Robust depth perception through Virtual Pattern Projection (VPP)**.  L. Bartolomei, M. Poggi, F. Tosi, A. Conti, S. Mattoccia; CVPR 2024 DEMO, Seattle, US [[Website](https://vppstereo.github.io/demo.html)] [[Code](https://github.com/bartn8/cvpr24-demo)] [[Flyer](https://vppstereo.github.io/assets/demo/DEMO_CVPR_2024_LARGE.pdf)]

## :fountain_pen: Citation

Please consider citing this list if you find this repository useful:

```bibtex
@article{Tosi_IJCV_2025,
  author    = {Fabio Tosi and Luca Bartolomei and Matteo Poggi},
  title     = {A Survey on Deep Stereo Matching in the Twenties},
  journal   = {International Journal of Computer Vision},
  year      = {2025},
  doi       = {10.1007/s11263-024-02331-0},
  url       = {https://doi.org/10.1007/s11263-024-02331-0},
  issn      = {1573-1405}
}
```

```bibtex
@article{poggi2021synergies,
  title={On the synergies between machine learning and binocular stereo for depth estimation from images: a survey},
  author={Poggi, Matteo and Tosi, Fabio and Batsos, Konstantinos and Mordohai, Philippos and Mattoccia, Stefano},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={9},
  pages={5314--5334},
  year={2021},
  publisher={IEEE}
}
```
```bibtex
@article{poggi2021confidence,
  title={On the confidence of stereo matching in a deep-learning era: a quantitative evaluation},
  author={Poggi, Matteo and Kim, Seungryong and Tosi, Fabio and Kim, Sunok and Aleotti, Filippo and Min, Dongbo and Sohn, Kwanghoon and Mattoccia, Stefano},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={44},
  number={9},
  pages={5293--5313},
  year={2021},
  publisher={IEEE}
}
```
