# PyVot : Python Variational Optimal Transportation
This package includes the prototype code for reproducing the map in Figure 4 (b) of the paper:

Mi, Liang, Wen Zhang, Xianfeng Gu, and Yalin Wang. "Variational Wasserstein Clustering." In Proceedings of the European Conference on Computer Vision (ECCV), pp. 322-337. 2018.

* Variational Wasserstein clustering in each iteration leverages variational principles [Gu et al. 2013] to solve optimal transportation. Thus, we name the package PyVot instead of PyVWc for the sake of pronunciation. For computing optimal transportation, simply set the max iteration to one.
* This program implements gradient descent instead of Newton's method to avoid computing convex hulls so that it can handle high-dimensional data. 

![alt text](data/sample.png?raw=true "Demo of variational Wasserstein clustering")

#### Notes
* Curvature is computed following [Ulen et al. 2015].

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

[2] Ulen, Johannes, Petter Strandmark, and Fredrik Kahl. "Shortest paths with higher-order regularization." IEEE transactions on pattern analysis and machine intelligence 37, no. 12 (2015): 2588-2600.

## Contact
Please contact Liang Mi icemiliang@gmail.com for any issues. 
