# Based Global Similar Image Patche and HOSVD for Image Denoising

### Cite this project


## Install
`git clone https://www.github.com/KazukiAmakawa/HOSVD_denoise_patch `

## Usage
### Easy Mode
Detemine the parameters in main.m and run the project

Parameter Table

|parameter          | Intro                                     |
|-------------------|:-----------------------------------------:|
|para_sigma         | Sigma of noise                            |
|para_betta         | Relaxation parameter (Learning Rate)      |
|para_gamma         | Scaling factor controlling                |
|para_patch_size    | Size of every patch                       |
|para_patch_stack   | Length of the tensor in SVD processing    |
|para_iteration     | Iteration times                           |
|test_switch        | Print the image on screen or not          |
|patch_method       | Patch Analysis method                     |

* We had found best para_betta and para_gamma in sigma = 10, 30 and 50. 

* We just trained the GMM pre-trained model in 7, 8, 9 and 10 patch size

* patch_method list

|Code               | Intro                                            |
|-------------------|:------------------------------------------------:|
|1                  | Original method NNM patch search                 |
|21/22              | Pre-trained Gaussian Mixture Model method only   |
|31/32/33/34/35     | Pre-trained GMM and K-means method               |
|21/31              | BFS after classification search                  |
|22/32              | Virtual reference patch method                   |
|33                 | All patch in class combined tensor               |
|34                 | Reference patch in classes to build tensor       |
|35                 | Double port search on image                      |



### Full Mode
You can using HOSVD_Denoising.m directly for your denoising.

Or, add new patch methods in Block_matching.m to test your new patch method with HOSVD

Source Code: https://www.github.com/KazukiAmakawa/HOSVD<br/>
For more help, submit issue in this project or connect this E-mail: GeorgeKahChen@gmail.com 


## LICENSE
Copyright (c) by KazukiAmakawa(Huayan Chen), all right reserved.<br/>
GNU GENERAL PUBLIC LICENSE Version 3

If you want to using this project as close source project, please connect us.



## Reference
[1] GARW: Group Attribute Random Walk: https://www.github.com/KazukiAmakawa/GARW-Class

 (This is a project working on non-linear classify method in deep learning with pytorch structure, still developing)

[2] F. Chen, L. Zhang and H. Yu, "External Patch Prior Guided Internal Clustering for Image Denoising," 2015 IEEE International Conference on Computer Vision (ICCV), Santiago, 2015, pp. 603-611. doi: 10.1109/ICCV.2015.76

 (We used GMM and K-means solution from code of this paper)

[3] R. Movchan and Z. Shen, "Adaptive thresholding hosvd algorithm with iterative regularization for image denoising," 2017 IEEE International Conference on Image Processing (ICIP), Beijing, 2017, pp. 2991-2995. doi: 10.1109/ICIP.2017.8296831

(We used most code in HOSVD from this paper)