## Demo

Below shows some examples of using PyVot.


#### A quick comparison to Sinkhorn OT
Suppose we want to align 100 blue source samples to another 100 red target samples. 
VOT produces a one-to-one map while Sinkhorn yields a 
many-to-may map due to the nature of these two algorithms. As a result,
for each source sample, Sinkhorn will give a weighted average location of its
multiple targets.

python vot_vs_sinkhorn.py

To run Sinkhorn OT, please install the [Python Optimal Transport](https://github.com/rflamary/POT) library, by e.g. pip install POT, 

![alt text](pics/vot_vs_sinkhorn.png?raw=true)
 

#### Area preserving mapping. 

![alt text](pics/area_preserve.png?raw=true)


#### Vector quantization by Wasserstein clustering.

![alt text](pics/color.png?raw=true)


#### Domain adaptation through Wasserstein clustering regularized by potential energy.

![alt text](pics/rwm_potential.png?raw=true)

#### Domain adaptation through Wasserstein clustering regularized by geometric transformation.

![alt text](pics/rwm_transform.png?raw=true)
