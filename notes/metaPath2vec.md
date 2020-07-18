# metapath2vec: Scalable Representation Learning for Heterogeneous Networks
> 论文：metapath2vec: Scalable Representation Learning for Heterogeneous Networks
> 作者：Yuxiao Dong  Microso Research
> 论文链接：https://www3.nd.edu/~dial/publications/dong2017metapath2vec.pdf
> 期刊：KDD 2017

今天看的这一篇是Yuxiao Dong发表在KDD2017的一篇关于异质图网络Graph Embedding。本文提出了基于元路径的随机游走来指定一个节点的邻居，之后利用异构skip-gram模型来实现embedding。

### 1、INTRODUCTION
传统的网络挖掘方法，一般通过将网络转化成邻接矩阵，在使用机器学习模型挖掘网络中的信息。但是，邻接矩阵通常都很稀疏，且维数很大。同时作者提到当前的一些**基于神经网络的模型针对复杂网络的表示学习**也有非常好的效果。其中包括当前已经提出的采用了word2vec思想的网络表示算法，如Deepwalk，node2vec以及LINE等。但是作者也明确指出了，上述这些算法虽然可以用于网络表示学习，但仅适合那些只包含一类顶点类型和边类型的同构网络（Homogeneous Networks），并不能很好地用于包含多种顶点类型和边类型的复杂关系网络。因此作者提出在基于meta-path的基础上，提出了在异构复杂关系网络的表示学习方法——metapath2vec和metapath2vec++。 metapath2vec的目标是最大化保留一个异构网络的结构和语义信息的似然，首先使用基于meta-path的随机游走获取异构网络中每种不同类型顶点的异构领域，然后使用扩展的Skip-Gram处理前面获取的顶点邻域，最终学习每个不同类型顶点的网络嵌入表示。

###2、 PROBLEM DEFINITION
 *  **Heterogeneous Network**
      异质网络定义为：$G=(V,E,T)$,其中每个节点和边的映射函数为：$ ϕ(v) : V → T_V 和  φ(e) : E → T_E ,
$ 其中Tv和TE分别表示定点和边的类型，并且满足$|T_V|+|T_E|>2$。

*  **Heterogeneous Network Representation Learning**
异构网络表征学习定义为：给定一个异构网络G，学习一个d维的潜在表征$X \in R^{|V| * d},d<<|V|$可以表征网络中顶点之间的结构信息和语义场景关系。模型的输出是一个低维的矩阵X，其中的第i行是一个d维的向量，表示定点i的表示。但是要注意一点，传统的同质图定点嵌入的表示特征方法很难直接应用于异质结构网络上。

###3、 Metapath2vec 
   在Metapath2vec 中，采用的方式和DeepWalk类似的方式，利用skip-gram来学习图的embedding。1、利用元路径随机游走从图中获取序列，2、利用skip-gram来学习节点的嵌入表示。

对于基于异构网络的metapath2vec嵌入算法，包含两个部分，分别是元路径随机游走(Meta-Path-Based Random Walks)和Heterogeneous Skip-Gram。

**Heterogeneous Skip-Gram**
在同构网络上的基于random walk的graph embedding算法通常对于一个同质网络$G=(V,E)$,目标是从每个顶点的局部邻域上最大化网络的似然：
$$argmax_\theta \prod_{v \in V} \prod_{c \in N(v)} p(c|v;\theta)$$
其中$N(v)$表示定点v的邻域，也就是其1-hop或2-hop的邻居节点。$ p(c|v;\theta)$表示在参数$\theta$下，给定节点v后，节点C的条件概率。

对于异质图$G=(V,E,T),|T_V|>1$,目标就是在给定节点v后，是的其上下文内容存在的概率最大化，如下：
$$argmax_\theta \sum_{v \in V} \sum_{t \in T_V} \sum_{c_t \in N_t(v)} p(c_t|v;\theta)$$
这里的$N_t(v)$指的是在节点v的邻近节点中，为第t个节点，而概率函数$p(c_t|v;\theta)$则为softmax。可表示为：$$p(c_t|v;\theta)=\frac{e^{X_{ct} . X_v}}{\sum_{u \in V} e^{X_u.X_v}}$$这里的$X_v$就是嵌入矩阵中的第v行向量，它表示节点v的嵌入向量。
metapath2vec中采用Negative Sampling进行参数迭代更新，这时设置一个负采样的窗口M，则参数更新过程如下：
![](https://upload-images.jianshu.io/upload_images/23885886-bba0f67367b3024b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中$\sigma(x)$是sigmoid函数，P(u)是一个negative node  $u^m$在M次采用中的预定义分布。

**metapath2vec通过不考虑顶点的类型进行节点抽取来确定当前顶点的频率。**

**Meta-Pathe-Based Random Walks**
metapath2vec采用的和deepwalk采用的是相同的思路，不过deepwalk处理的是同质图，但是在异质图中决定下一步随机游走的条件概率$p(v^{i+1}| v^i)$如果只对节点vi的邻居节点做标准化，而不对节点类型进行考虑，在其他论文中证明出：**异质网络上的随机游走生成的路径，偏向（biased）于高度可见的节点类型（具有优势/主导数量的路径的节点）和 集中(concentrated)的节点（即：具有指向一小组节点路径的 大部分百分比）。**
于是本文提出基于元路径的随机游走，获取不同节点之间的语义及结构相关性。
基于元路径的随机游走可以定义成如下形式：
![](https://upload-images.jianshu.io/upload_images/23885886-075d6e1cc46efed8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中
![](https://upload-images.jianshu.io/upload_images/23885886-694af9d0170686c9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
表示节点v1和节点vl之间的路径组合。
举个例子，下图中“APA”表示一种固定语义的meta-path，“APVPA”表示另外一种固定语义的meta-path。而这样的meta-path可以帮助挖掘网络中更多的信息，因此，在本文中，作者给出了基于meta-path的随机游走方式。
![](https://upload-images.jianshu.io/upload_images/23885886-e9722068a22d600a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

给定一个异质网络图$G=(V,E,T)$和一个meta-path的模板(scheme) $\mathcal{p}$，那么在第i步的转移概率定义图如下：
$$\begin{equation}
p(v^{i+1} | v^i_t;\mathcal{P})= \left \{
\begin{array}{ll}
\frac{1}{N_{t+1}(v^i_t)}  (v^{i+1},v^i_t) \in E, \phi(v^{i+1})=t+1\\
0  \quad\quad\quad (v^{i+1},v^i_t) \in E, \phi(v^{i+1}) \neq t+1\\
0 	\quad\quad\quad\quad (v^{i+1},v^i_t) \notin E\\
\end{array}\right.
\end{equation}$$

其中$v^i_t∈V_t$, 并且$N_{t+1}(v^i_t)$代表的是节点v^i_t的邻居中属于t+1type的节点集合。换句话说，游走是在预先设定的meta-path $\mathcal{p}$的条件上。而且，meta-path一般都是用在对称的路径上,也就是说在上述路径组合中，顶点$V_t$的类型和$V_l$的类型相同。

$\begin{equation}
p(v^{i+1}|v_t^i) = p(v^{i+1}|v_l^i),\quad  if \quad t=l
\end{equation}$

基于meta-path的随机游走保证不同类型顶点之间的语义关系之后，可以适当的融入Skip-Gram模型中进行训练得到节点的嵌入表示。

###4、 Metapath2vec++
由于在meta-path中我们是根据节点的类型进行的随机游走，但是在在softmax环节中，我们是将所有节点按照同一种类型进行的负采样过程，并未按照节点的类型进行区分，也就是说metapath2vec支持任意类型顶点的Negative Sampling。于是就与这一点作者进行了改进提出了Metapath2vec++。

因而本文提出，异质的负采样(Heterogeneous negative sampling)。也就是说softmax函数根据节点的不同类型进行归一化处理，那么$p(c_t|v; \theta)$是根据固定类型的顶点进行调整。即：
![](https://upload-images.jianshu.io/upload_images/23885886-06a6f02ea734e4ff.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
这同时为了skip-gram最后一层输出层中的 每个类型都指定了一个多项分布。负采样的目标函数：
![](https://upload-images.jianshu.io/upload_images/23885886-88744ce6a931d2a3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 5、Conclusion
本篇论文继续沿用了同构图上基于随机游走的Embedding算法的思想，不过通过meta-path来指导生产随机游走的过程，使得在异质图中的异构信息和语义信息保留，同时借助Skip-Gram模型可以学习节点的表征。


####参考
1.[metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://www3.nd.edu/~dial/publications/dong2017metapath2vec.pdf)

2.[Graph Embedding之metapath2vec](https://www.jianshu.com/p/b34a0703eb89)










