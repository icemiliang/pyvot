# PyVot : Python Variational Optimal Transportation
This package includes the prototype code for computing variational Wasserstein clustering.

* Variational Wasserstein clustering in each iteration leverages variational principles [Gu et al. 2013] to solve optimal transportation. Thus, we name the package PyVot instead of PyVWc for the sake of pronunciation. For computing optimal transportation, simply set the max iteration to one.
* This program implements gradient descent instead of Newton's method to avoid computing convex hulls so that it can handle high-dimensional data. 
* The picture below shows the results from regularized Wasserstein clustering.
![alt text](data/rwm.png?raw=true "Demo of regularized Wasserstein means")


## Dependencies
* Python >= 3.5
* NumPy >= 1.15.4
* SciPy >= 1.1.0
* Matplotlib >= 3.0.2 (for demo only)

## References
#### Citing the package

If you find the code helpful, please cite the following article:

Mi, Liang, Wen Zhang, Xianfeng Gu, and Yalin Wang. "Variational Wasserstein Clustering." In Proceedings of the European Conference on Computer Vision (ECCV), pp. 322-337. 2018.
```
@inproceedings{mi2018variational,
  title={Variational {W}asserstein Clustering},
  author={Mi, Liang and Zhang, Wen and Gu, Xianfeng and Wang, Yalin},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={322--337},
  year={2018}
}
```
#### Additional references

[1] Gu, Xianfeng, Feng Luo, Jian Sun, and S-T. Yau. "Variational principles for Minkowski type problems, discrete optimal transport, and discrete Monge-Ampere equations." arXiv preprint arXiv:1302.5472 (2013).

## Contact
Please contact Liang Mi icemiliang@gmail.com for any issues. 
