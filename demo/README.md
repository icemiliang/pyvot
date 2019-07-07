## Demo

Below shows some examples of using PyVot.


#### A quick comparison to Sinkhorn OT
Suppose we want to align 100 samples to another 100 samples. 
VOT produces a one-to-one map while Sinkhorn yields a 
many-to-may map due to the nature of these two maps. As a result,
for each source sample, Sinkhorn will give a weighted average location of its
multiple targets. VOT constantly consumes less CPU time than Sinkhorn (only for this example). 
A more comprehensive comparison between VOT and Sinkhorn OT is a todo.

python vot_vs_sinkhorn.py

To run Sinkhorn OT, please install the [Python Optimal Transport](https://github.com/rflamary/POT) library, by e.g. pip install POT, 

![alt text](pics/vot_vs_sinkhorn.png?raw=true)
 

#### Area preserving mapping. 

python demo_area_preserve.py

![alt text](pics/area_preserve.png?raw=true)
![alt text](pics/area_preserve.gif?raw=true)

#### Domain adaptation through Wasserstein clustering regularized by potential energy.

python demo_potential.py

![alt text](pics/rwm_potential.png?raw=true)

#### Domain adaptation through Wasserstein clustering regularized by geometric transformation.

python demo_transform.py

![alt text](pics/rwm_transform.png?raw=true)
