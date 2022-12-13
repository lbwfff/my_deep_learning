# for_my_deep_learning

最近在学keras什么的，一边在折腾python一边在学习深度学习算法（主要是CNN什么的）,然后正好有google colab这样一个平台嘛，应该好好的应用起来

## a_demo_for_peptide_pl  

a_demo_for_peptide_pl,是一段我在google colab上写的代码，使用肽的序列信息预测等电点的回归模型。使用了包括one-hot化的序列信息，一批氨基酸信息（12个，乱七八糟的），AAC氨基酸组成信息，汇聚成3X50X50的三维矩阵（我只考虑学习和预测50个氨基酸长度以下的肽）

矩阵的构成我还是很满意的，感觉循环什么的，因为有R语言的基础还是很好理解的，只是很多python的基础函数需要边用边查。模型的表现的话，因为google colad服务器的内存问题，我一直没法做太多的操作（其实还是一个python能力的问题，我记得书上有提到要怎么缓解内存的压力，但是自己一直没有很好的理解）。但在使用不到十分之一的数据做训练和测试情况下得到了还算不错的一个表现。

后续的设想就是，想使用更加复杂的网络来实现，比如残差网络块啊什么的，更加现代一些的深度学习模块，把显卡算力给用上（现在的网络用CPU训练都不算慢，确实是一个非常简单的网络）。想要做到这一点，首先是对于理论的知识，然后对于代码的实现，都是需要学习的，路漫漫其修远兮，但至少很有趣。

## my_learning_11-29

我重新做了数组，之前对于CNN维度的理解是错的，数据应该往第三轴堆积。

然后建立了残差网络的模型，在低迭代次数的情况下残差网络模型好像没有比简单CNN表现更好，我不确定更多的迭代次数会不会让表现比简单CNN更好。

ps:100次迭代的话mae可以到0.127的状态，已经接近IPC2.0的表现了，我没跑完200次迭代但感觉大差不差就这个状态了，深度学习果然是一个大力出奇迹的东西

#中间这段时间在给实验室做组学数据，没有时间折腾

## Biological_Image.ipynb

随手做了一个图像分类的模型，数据来自Cai, L., Wang, Z., Kulathinal, R., Kumar, S., & Ji, S. (2021). Deep Low-Shot Learning for Biological Image Classification and Visualization From Limited Training Samples. IEEE transactions on neural networks and learning systems, PP, 10.1109/TNNLS.2021.3106831. Advance online publication. https://doi.org/10.1109/TNNLS.2021.3106831

简单用CNN训练了一下，ACC直接到1了，然后我都不知道要怎么继续往下做了
