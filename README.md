# PyVot : Python Variational Optimal Transportation
This package includes the prototype code for computing variational optimal transportation and Wasserstein clustering.

* Variational Wasserstein clustering in each iteration leverages variational principles [1] to solve optimal transportation. Thus, we name the package PyVot instead of PyVWc for the sake of pronunciation. For computing optimal transportation, simply set the max iteration to one.
* This program implements gradient descent instead of Newton's method to avoid computing convex hulls so that it can handle high-dimensional data. 
* The picture below shows the results from original Wasserstein clustering [2] and regularized Wasserstein clustering [3].
* Below shows two examples.

demo_potential.py
![alt text](data/rwm_potential.png?raw=true)

demo_transform.py
![alt text](data/rwm_transform.png?raw=true)

## Dependencies
* Python >= 3.5
* NumPy >= 1.15.4
* SciPy >= 1.1.0 (for regularization)
* scikit-image >= 0.14.1 (for geometric transformation)
* Matplotlib >= 3.0.2 (for demo)

The code has been tested on Ubuntu 16.04 and macOS 10.14

## References
#### Citing the package

If you find the code helpful, please cite one of the following articles:

**Citing the original Wasserstein clustering:**

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

**Citing the regularized Wasserstein clustering:**

Mi, Liang, Wen Zhang, and Yalin Wang. "Regularized Wasserstein Means Based on Variational Transportation." arXiv preprint arXiv:1812.00338 (2018).
```
@article{mi2018regularized,
  title={Regularized Wasserstein Means Based on Variational Transportation},
  author={Mi, Liang and Zhang, Wen and Wang, Yalin},
  journal={arXiv preprint arXiv:1812.00338},
  year={2018}
}
```

#### Additional references

[1] Gu, Xianfeng, Feng Luo, Jian Sun, and S-T. Yau. "[Variational principles for Minkowski type problems, discrete optimal transport, and discrete Monge-Ampere equations.](https://arxiv.org/abs/1302.5472)" arXiv preprint arXiv:1302.5472 (2013).

[2] Mi, Liang, Wen Zhang, Xianfeng Gu, and Yalin Wang. "[Variational Wasserstein Clustering.](https://arxiv.org/abs/1806.09045)" In Proceedings of the European Conference on Computer Vision (ECCV), pp. 322-337. 2018.

[3] Mi, Liang, Wen Zhang, and Yalin Wang. "[Regularized Wasserstein Means Based on Variational Transportation.](http://arxiv.org/abs/1812.00338)" arXiv preprint arXiv:1812.00338 (2018).

## Contact
Please contact Liang Mi icemiliang@gmail.com for any issues. 
