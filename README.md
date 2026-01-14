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


## Citation 
If you use this codebase, or otherwise find our work valuable, please cite Skimba:

```
@article{skimba_2025, 
    title={Skip Mamba Diffusion for Monocular 3D Semantic Scene Completion}, 
    volume={39}, 
    url={https://ojs.aaai.org/index.php/AAAI/article/view/32547}, 
    DOI={10.1609/aaai.v39i5.32547}, 
    number={5}, 
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    author={Liang, Li and Akhtar, Naveed and Vice, Jordan and Kong, Xiangrui and Mian, Ajmal Saeed}, 
    year={2025}, 
    month={Apr.}, 
    pages={5155-5163} 
}
```