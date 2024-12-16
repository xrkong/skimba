# Skip Mamba Diffusion for Monocular 3D Semantic Scene Completion (AAAI 2025)

Li Liang<sup>1</sup>, Naveed Akhtar<sup>2</sup>, Jordan Vice<sup>1</sup>, Xiangrui Kong<sup>1</sup>, Ajmal Mian<sup>1</sup>,

<sup>1</sup>The University of Western Australia  
<sup>2</sup>The University of Melbourne

![Figure_1](https://github.com/user-attachments/assets/5dbac60c-8999-4312-abd8-8570fecf6ffa)
*Figure 1: Schematics of the approach. Our method comprises a 3D scene completion and a 3D semantic segmentation network.
The former is encapsulated in a VAE framework that employs two sub-networks for conditioning its latent space, a Muti-Scale
Convolutonal Block (MSCB) and a Skimba denoising network. The 3D semantic segmentation network employs a variant of
Skimba. L, W, and H denote the length, width, and height of the original scene, and D is feature map dimension.*  
  
![Figure_2](https://github.com/user-attachments/assets/34f2bdd6-a0ce-46f7-8c6a-5c5a8474dacc)
*Figure 2: Architectural details of the Skimba denoising network. Refer to the text for details.*  
  
![Figure_3](https://github.com/user-attachments/assets/2ed4fbdf-92d4-4f23-96f6-e7fe210c0554)
*Figure 3: Qualitative results on the SemanticKITTI validation set. Columns from the left represent, input data, ground truth,
and outputs of SkimbaDiff (our method), MonoScene, OccFormer, and VoxFormer-T (a stereo method)*  

We will release the code soon.
