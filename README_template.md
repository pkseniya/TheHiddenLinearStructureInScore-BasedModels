# The Hidden Linear Structure in Score-Based Models and its Application

## Discription of our project

In our project we recreated methods described in the [paper](https://arxiv.org/pdf/2311.10892.pdf) about accelerating sampling process from diffusion models. 
Score-based models have achieved remarkable results in the generative modeling of many domains. 
By learning the gradient of smoothed data distribution, we can iteratively generate samples from complex distribution e.g. natural images. 
However, is there any universal structure in the gradient field that will eventually be learned by any neural network? 
In the paper the authors aim to find such structures through a normative analysis of the score function. 
First, they derived the closed-form solution to the scored-based model with a Gaussian score. 
The authors claimed that for well-trained diffusion models, the learned score at a high noise scale is well approximated by the linear score of Gaussian. 
They demonstrated this through empirical validation of pre-trained images diffusion model and theoretical analysis of the score function. 
Their finding of the linear structure in the score-based model has implications for better model design and data pre-processing.


## Methods (short description of our project)

- 
- 
- 
- 


## Getting started

To reproduce the main results from our paper, just run:

```
python generate.py
```



## Developers 

[Daniil Shlenskii](https://github.com/daniil-shlenskii)

[Kseniia Petrushina](https://github.com/pkseniya)

[Nikita Kornilov]()

[Bair Mikhailov](https://github.com/MikhailovBair)

[Arina Chumachenko](https://github.com/arina-chumachenko)
