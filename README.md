# 3D Shape Reconstruction From A Single 2D Image
## I. Paper Research

1. [Re-imagine the Negative Prompt Algorithm: Transform 2D Diffusion into 3D, alleviate Janus problem and Beyond](https://arxiv.org/pdf/2304.04968v3.pdf)

    _11/04/2023_
    
    _Authors: Mohammadreza Armandpour, Ali Sadeghian, Huangjie Zheng, Amir Sadeghian, Mingyuan Zhou_
      
   _link paper: https://arxiv.org/pdf/2304.04968v3.pdf_

   1. ### Abstract:
      
      Although text-to-image diffusion models have made significant strides in generating images from text, they are sometimes more inclined to generate images like the data on which the model was trained rather than the provided text. This limitation has hindered their usage in both 2D and 3D applications. To address this problem, we explored the use of negative prompts but found that the current implementation fails to produce desired results, particularly when there is an overlap between the main and negative prompts. To overcome this issue, we propose Perp-Neg, a new algorithm that leverages the geometrical properties of the score space to address the shortcomings of the current negative prompts algorithm. Perp-Neg does not require any training or fine-tuning of the model. Moreover, we experimentally demonstrate that Perp-Neg provides greater flexibility in generating images by enabling users to edit out unwanted concepts from the initially generated images in 2D cases. Furthermore, to extend the application of Perp-Neg to 3D, we conducted a thorough exploration of how Perp-Neg can be used in 2D to condition the diffusion model to generate desired views, rather than being biased toward the canonical views. Finally, we applied our 2D intuition to integrate Perp-Neg with the state-of-the-art text-to-3D (DreamFusion) method, effectively addressing its Janus (multi-head) problem
   2. ### Code:
      link git: https://github.com/ashawkey/stable-dreamfusion
   
      Quickstart with colab :  https://colab.research.google.com/drive/1MXT3yfOFvO0ooKEfiUUvTKwUkrrlCHpF?usp=sharing
2. [DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/pdf/2209.14988v1.pdf)
   
   _29/09/2022_
    
    _Authors: Ben Poole, Ajay Jain, Jonathan T. Barron, Ben Mildenhall_
      
   _link paper: https://arxiv.org/pdf/2209.14988v1.pdf_

   1. ### Abstract:
   
      Recent breakthroughs in text-to-image synthesis have been driven by diffusion models trained on billions of image-text pairs. Adapting this approach to 3D synthesis would require large-scale datasets of labeled 3D data and efficient architectures for denoising 3D data, neither of which currently exist. In this work, we circumvent these limitations by using a pretrained 2D text-to-image diffusion model to perform text-to-3D synthesis. We introduce a loss based on probability density distillation that enables the use of a 2D diffusion model as a prior for optimization of a parametric image generator. Using this loss in a DeepDream-like procedure, we optimize a randomly-initialized 3D model (a Neural Radiance Field, or NeRF) via gradient descent such that its 2D renderings from random angles achieve a low loss. The resulting 3D model of the given text can be viewed from any angle, relit by arbitrary illumination, or composited into any 3D environment. Our approach requires no 3D training data and no modifications to the image diffusion model, demonstrating the effectiveness of pretrained image diffusion models as priors.
   2. ### Code:
      link git: 
      * https://github.com/ashawkey/stable-dreamfusion
      * https://github.com/chinhsuanwu/dreamfusionacc
      * https://github.com/SusungHong/IF-DreamFusion
      * https://github.com/muelea/buddi
      
3. [Deep3D: Fully Automatic 2D-to-3D Video Conversion with Deep Convolutional Neural Networks](https://arxiv.org/pdf/1604.03650v1.pdf)
   
   _13/04/2016_
    
   _Authors: Junyuan Xie, Ross Girshick, Ali Farhadi_
      
   _link paper: https://arxiv.org/pdf/1604.03650v1.pdf_
      
   1. ### Abstract:
   
      As 3D movie viewing becomes mainstream and Virtual Reality (VR) market emerges, the demand for 3D contents is growing rapidly. Producing 3D videos, however, remains challenging. In this paper we propose to use deep neural networks for automatically converting 2D videos and images to stereoscopic 3D format. In contrast to previous automatic 2D-to-3D conversion algorithms, which have separate stages and need ground truth depth map as supervision, our approach is trained end-to-end directly on stereo pairs extracted from 3D movies. This novel training scheme makes it possible to exploit orders of magnitude more data and significantly increases performance. Indeed, Deep3D outperforms baselines in both quantitative and human subject evaluations.
   
   2. ### Code:
      link git:
         * https://github.com/piiswrong/deep3d
         * https://github.com/LouisFoucard/w-net
         * https://github.com/Candice-X/w-net-for-image-segmentation
         * https://github.com/pesuchin/Deep3D-chainer
   
4. [Magic123: One Image to High-Quality 3D Object Generation Using Both 2D and 3D Diffusion Priors](https://arxiv.org/pdf/2306.17843v2.pdf)
   
   _30/06/2023_
   
   _Authors: Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, Bernard Ghanem_

   _link paper: https://arxiv.org/pdf/2306.17843v2.pdf_

   1. ### Abstract:
   
      We present Magic123, a two-stage coarse-to-fine approach for high-quality, textured 3D meshes generation from a single unposed image in the wild using both2D and 3D priors. In the first stage, we optimize a neural radiance field to produce a coarse geometry. In the second stage, we adopt a memory-efficient differentiable mesh representation to yield a high-resolution mesh with a visually appealing texture. In both stages, the 3D content is learned through reference view supervision and novel views guided by a combination of 2D and 3D diffusion priors. We introduce a single trade-off parameter between the 2D and 3D priors to control exploration (more imaginative) and exploitation (more precise) of the generated geometry. Additionally, we employ textual inversion and monocular depth regularization to encourage consistent appearances across views and to prevent degenerate solutions, respectively. Magic123 demonstrates a significant improvement over previous image-to-3D techniques, as validated through extensive experiments on synthetic benchmarks and diverse real-world images.
   
   2. ### Code:
      
      link git: https://github.com/guochengqian/magic123
   
5. [SceneDreamer: Unbounded 3D Scene Generation from 2D Image Collections](https://arxiv.org/pdf/2302.01330v2.pdf)
   
   _02/02/2023_

   _Authors: Zhaoxi Chen, Guangcong Wang, Ziwei Liu_

   _link paper: https://arxiv.org/pdf/2302.01330v2.pdf_

   1. ### Abstract:
   
      In this work, we present SceneDreamer, an unconditional generative model for unbounded 3D scenes, which synthesizes large-scale 3D landscapes from random noise. Our framework is learned from in-the-wild 2D image collections only, without any 3D annotations. At the core of SceneDreamer is a principled learning paradigm comprising 1) an efficient yet expressive 3D scene representation, 2) a generative scene parameterization, and 3) an effective renderer that can leverage the knowledge from 2D images. Our approach begins with an efficient bird's-eye-view (BEV) representation generated from simplex noise, which includes a height field for surface elevation and a semantic field for detailed scene semantics. This BEV scene representation enables 1) representing a 3D scene with quadratic complexity, 2) disentangled geometry and semantics, and 3) efficient training. Moreover, we propose a novel generative neural hash grid to parameterize the latent space based on 3D positions and scene semantics, aiming to encode generalizable features across various scenes. Lastly, a neural volumetric renderer, learned from 2D image collections through adversarial training, is employed to produce photorealistic images. Extensive experiments demonstrate the effectiveness of SceneDreamer and superiority over state-of-the-art methods in generating vivid yet diverse unbounded 3D worlds.
   
   2. ### Code:
   
      link git: https://github.com/frozenburning/scenedreamer

      faster demo with Hugging Face : https://huggingface.co/spaces/FrozenBurning/SceneDreamer
      
   
6. [NeuralLift-360: Lifting An In-the-wild 2D Photo to A 3D Object with 360° Views](https://arxiv.org/pdf/2211.16431v2.pdf)

   _29/11/2022_
   
   _Authors: Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Yi Wang, Zhangyang Wang_
   
   _link paper: https://arxiv.org/pdf/2211.16431v2.pdf_

   1. ### Abstract:
   
      Virtual reality and augmented reality (XR) bring increasing demand for 3D content. However, creating high-quality 3D content requires tedious work that a human expert must do. In this work, we study the challenging task of lifting a single image to a 3D object and, for the first time, demonstrate the ability to generate a plausible 3D object with 360{\deg} views that correspond well with the given reference image. By conditioning on the reference image, our model can fulfill the everlasting curiosity for synthesizing novel views of objects from images. Our technique sheds light on a promising direction of easing the workflows for 3D artists and XR designers. We propose a novel framework, dubbed NeuralLift-360, that utilizes a depth-aware neural radiance representation (NeRF) and learns to craft the scene guided by denoising diffusion models. By introducing a ranking loss, our NeuralLift-360 can be guided with rough depth estimation in the wild. We also adopt a CLIP-guided sampling strategy for the diffusion prior to provide coherent guidance. Extensive experiments demonstrate that our NeuralLift-360 significantly outperforms existing state-of-the-art baselines
   
   2. ### Code:
      
      link git: https://github.com/VITA-Group/NeuralLift-360
      
      Quickstart with colab: https://colab.research.google.com/drive/15YCsqaO6l94HueVwPQgHqVVDUJzdOEO5?usp=sharing
         
7. [Pose2Mesh: Graph Convolutional Network for 3D Human Pose and Mesh Recovery from a 2D Human Pose](https://arxiv.org/pdf/2008.09047v3.pdf)

   _ECCV 2020_

   _Authors: Hongsuk Choi, Gyeongsik Moon, Kyoung Mu Lee_

   _link paper : https://arxiv.org/pdf/2008.09047v3.pdf_

   1. ### Abstract:
   
      Most of the recent deep learning-based 3D human pose and mesh estimation methods regress the pose and shape parameters of human mesh models, such as SMPL and MANO, from an input image. The first weakness of these methods is an appearance domain gap problem, due to different image appearance between train data from controlled environments, such as a laboratory, and test data from in-the-wild environments. The second weakness is that the estimation of the pose parameters is quite challenging owing to the representation issues of 3D rotations. To overcome the above weaknesses, we propose Pose2Mesh, a novel graph convolutional neural network (GraphCNN)-based system that estimates the 3D coordinates of human mesh vertices directly from the 2D human pose. The 2D human pose as input provides essential human body articulation information, while having a relatively homogeneous geometric property between the two domains. Also, the proposed system avoids the representation issues, while fully exploiting the mesh topology using a GraphCNN in a coarse-to-fine manner. We show that our Pose2Mesh outperforms the previous 3D human pose and mesh estimation methods on various benchmark datasets.
   
   2. ### Code:
   
      link git: 
   
         * https://github.com/hongsukchoi/Pose2Mesh_RELEASE
         * https://github.com/karanshahgithub/CS256-AI-Pose2Mesh
            
8. [DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation](https://arxiv.org/pdf/2309.16653.pdf)

   _28/09/2023_

   _Authors: Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, Gang Zeng_

   _link paper: https://arxiv.org/pdf/2309.16653.pdf

   1. ### Abstract:

      Recent advances in 3D content creation mostly leverage optimization-based 3D generation via score distillation sampling (SDS). Though promising results have been exhibited, these methods often suffer from slow per-sample optimization, limiting their practical usage. In this paper, we propose DreamGaussian, a novel 3D content generation framework that achieves both efficiency and quality simultaneously. Our key insight is to design a generative 3D Gaussian Splatting model with companioned mesh extraction and texture refinement in UV space. In contrast to the occupancy pruning used in Neural Radiance Fields, we demonstrate that the progressive densification of 3D Gaussians converges significantly faster for 3D generative tasks. To further enhance the texture quality and facilitate downstream applications, we introduce an efficient algorithm to convert 3D Gaussians into textured meshes and apply a fine-tuning stage to refine the details. Extensive experiments demonstrate the superior efficiency and competitive generation quality of our proposed approach. Notably, DreamGaussian produces high-quality textured meshes in just 2 minutes from a single-view image, achieving approximately 10 times acceleration compared to existing methods.
   
   2. ### Code:
   
      link git:
         * https://github.com/dreamgaussian/dreamgaussian
         * https://github.com/camenduru/dreamgaussian-colab
      
      Quickstart with colab: https://colab.research.google.com/drive/1sLpYmmLS209-e5eHgcuqdryFRRO6ZhFS?usp=sharing

      faster demo with Huggingface : https://huggingface.co/spaces/jiawei011/dreamgaussian