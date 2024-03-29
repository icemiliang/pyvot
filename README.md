# PyVot : Python Variational Optimal Transportation
This is a collection of code for computing semi-discrete Monge optimal transportation (OT) through a variational method.

We named our method *variational optimal transportation* (VOT) or *variational Wasserstein clustering* (VWC).

* Given the empirical distributions (marginals) and the initial centroids, 
the vinilla VWC compute VOT and then update the centroids to the centers of mass.
The whole process will converge in one iteration. Our method is immune to unbalance measures.

* When there are two or more marginals, 
we are computing a discrete Wasserstein barycenter, usually with free support and fixed 
measure to simplify the problem. When there are two marginals, the results can
be used to solve the optimal transshipment problem.
  
* Monge OT maps exist in general when the marginals are continuous. 
In practice, we consider a collection of dense Dirac samples as an approximation.
  
* Our OT formulation is convex. Yet, this program only uses the 1st order gradient
 because the 2nd order gradient involves computing convex hulls which is intractable 
 for high-dimensional data.
 
* The picture below shows 10 random nested ellipses averaged according to the 
Euclidean distance (left) and the Wasserstein distance (right). More examples 
can be found in [demo/](demo/).

<img src="demo/rings/rings.png" width="50%">

## Dependencies

PyVot was implemented with both NumPy and PyTorch. It requires the following libraries: 

numpy, scipy, imageio, scikit-image, scikit-learn, Matplotlib, PyTorch (optional), POT (optional)

Install prerequisites via conda:
```
conda env create -f environment.yml
```

## References

#### Citing the package

If you find the code helpful, please cite one of the following articles:

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

Mi, Liang, Wen Zhang, and Yalin Wang. "Regularized Wasserstein means for aligning distributional data." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 04, pp. 5166-5173. 2020.
```
@inproceedings{mi2020regularized,
  title={Regularized Wasserstein means for aligning distributional data},
  author={Mi, Liang and Zhang, Wen and Wang, Yalin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={04},
  pages={5166--5173},
  year={2020}
}
```

#### Additional references

[1] Gu, Xianfeng, Feng Luo, Jian Sun, and S-T. Yau. "[Variational principles for Minkowski type problems, discrete optimal transport, and discrete Monge-Ampere equations.](https://arxiv.org/abs/1302.5472)" arXiv preprint arXiv:1302.5472 (2013).

[2] Mi, Liang, Wen Zhang, Xianfeng Gu, and Yalin Wang. "[Variational Wasserstein Clustering.](https://arxiv.org/abs/1806.09045)" In Proceedings of the European Conference on Computer Vision (ECCV), pp. 322-337. 2018.

[3] Mi, Liang, Wen Zhang, and Yalin Wang. "[Regularized Wasserstein means for aligning distributional data.](https://ojs.aaai.org/index.php/AAAI/article/view/5960)" In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 04, pp. 5166-5173. 2020.

[4] Mi, Liang. "[Variational wasserstein barycenters for geometric clustering.](https://arxiv.org/abs/2002.10543)" arXiv preprint arXiv:2002.10543 (2020).
 

## Contact

Please contact Liang Mi (icemiliang@gmail.com) for any issues. Pull requests and issues are welcome.
