## 第3章 线性模型

### 3.1 基本形式

​		线性模型试图学得一个通过属性的==线性组合==来进行预测的函数，即
$$
f(x)=w_{1}x_{1}+w_{2}x_{2}+...+w_{d}x_{d}+b\tag{3.1}
$$
一般用向量形式写成
$$
f(x)=\omega^{T}x+b\tag{3.2}
$$
其中$\omega=(w_{1},w_{2},...,w_{d})$. $\omega$ 和 $b$ 学得之后，模型就得以确定。

###### 线性模型的优点：

* 形式简单，易于建模；
* 具有很好的可解释性；
  * $\omega$ 直观表达了各属性在预测中的重要性

### 3.2 线性回归

​		线性回归视图学得一个线性模型以尽可能准确地预测实值输出标记。

​		确定$\omega,b$的关键在于如何衡量$f(x)$与$y$之间的差别，均方误差是最常用的性能度量，即
$$
\begin{aligned}
(\omega^{*},b^{*})&=argmin_{(\omega,x)}\sum_{i=1}^{m}(f(x_{i})-y_{i})^{2}\\
&=argmin_{(\omega,x)}\sum_{i=1}^{m}(y_{i}-wx_{i}-b)^{2}\\
\end{aligned}\tag{3.4}
$$
​		基于均方误差最小化来进行模型求解的方法称为“最小二乘法”。在线性回归中，最小二乘法就是试图找到一条直线，使得所有样本到直线上的欧式距离之和最小。

​		求解$\omega,b$使$E_{(w,b)}=\sum_{i=1}^{m}(y_{i}-wx_{i}-b)^{2}$最小化的过程，称为线性回归模型的最小二乘==“参数估计”==。$E_{(w,b)}$对$\omega,b$分别求导得到：
$$
\begin{aligned}
\frac{\part E_{w,b}}{\part w}=2(w\sum_{i=1}^{m}x_{i}^{2}-\sum_{i=1}^{m}(y_{i}-b)x_{i})
\end{aligned}\tag{3.5}\\
$$

$$
\begin{aligned}
\frac{\part E_{w,b}}{\part b}=2(mb-\sum_{i=1}^{m}(y_{i}-b))
\end{aligned}\tag{3.6}
$$

​		令（3.5）（3.6）为0可得$\omega,b$最优的闭式解
$$
w = \frac{\sum_{i=1}^{m}y_{i}(x_{i}-\bar{x})}{\sum_{i=1}^{m}x_{i}^{2}-\frac{1}{m}(\sum_{i=1}^{m})^2}\\\tag{3.7}
$$

$$
b=\frac{1}{m}\sum_{i=1}^{m}(y_{i}-wx_{i})\\\tag{3.8}
$$

其中，$\bar{x}=\frac{1}{m}\sum_{i=1}^{m}x_{i}$为$x$的均值。

​		延伸而来，
$$
f(x_{i})=\omega^{T}x_{i}+b,使得f(x_{i}\simeq y_{i})
$$
这被称为多元线性回归。

​		类似的，可利用最小二乘法对$w,b$进行估计。令$\hat{\boldsymbol{w}}=(w,b)$，数据集$D$表示为一个$m \times (d+1)$的矩阵$X$：
$$
\boldsymbol{X}=\left(
\begin{matrix}
x_{11} & x_{11} &\cdots & x_{1d} &1\\
x_{21} & x_{22} &\cdots & x_{2d} &1 \\
\vdots & \vdots &\cdots & \vdots &\vdots\\
x_{m1} & x_{m2} &\cdots & x_{md} &1\\
\end{matrix}
\right)=\left(
\begin{matrix}
x_{1}^{T} & 1\\
x_{2}^{T} & 1\\
\vdots & \vdots\\
x_{m}^{T} & 1\\
\end{matrix}
\right)
$$
再把标记写成向量形式$\boldsymbol{y}=(y_{1};y{2};...;y_{m})$，则有
$$
\hat{\boldsymbol{w}}^{*}=argmin_{\hat{\boldsymbol{w}}}(\boldsymbol{y}-\boldsymbol{X\hat{w}})^{T}(\boldsymbol{y}-\boldsymbol{X\hat{w}})\tag{3.9}
$$
令$E_{\hat{w}}=(\boldsymbol{y}-\boldsymbol{X\hat{w}})^{T}(\boldsymbol{y}-\boldsymbol{X\hat{w}})$，对$\hat{\boldsymbol{w}}$求导可得：
$$
\frac{\part E_{\hat{\boldsymbol{w}}}}{\part \hat{\boldsymbol{w}}}=2\boldsymbol{X}^{T}(\boldsymbol{X} \hat{\boldsymbol{w}}-\boldsymbol{y})\tag{3.10}
$$
令上式为0即为$\boldsymbol{\hat{w}}$最优解的闭式解。

​		(==没看懂==)做简单讨论：

​		当$\boldsymbol{X}^{T}\boldsymbol{X}$为满秩矩阵或者正定矩阵时，令（3.10）为0可得：
$$
\boldsymbol{\hat{w}^*}=(\boldsymbol{X}^{T}\boldsymbol{X})^{-1}\boldsymbol{X}^{T}\boldsymbol{y}\tag{3.11}
$$
其中，$(\boldsymbol{X}^{T}\boldsymbol{X})^{-1}$是$(\boldsymbol{X}^{T}\boldsymbol{X}）$的逆矩阵，令$\boldsymbol{\hat{x}_{i}}=(\boldsymbol{x_{i}},1)$，则最终学得的多元线性回归模型为：
$$
f(\hat{x_{i}})=\hat{x}_{i}^{T}(\boldsymbol{X}^{T}\boldsymbol{X})^{-1}\boldsymbol{X}^{T}\boldsymbol{y}\tag{3.12}
$$
​		但现实任务中，$(\boldsymbol{X}^{T}\boldsymbol{X})$往往不是满秩矩阵（例如变量数超过样例数），此时可以解出多个$\boldsymbol{\hat{w}}$，均满足均方误差最小化的要求，对于解的选择将由学习算法的归纳偏好决定，常见的做法是引入==正则化==项。

​		广义的线性模型：
$$
y=g^{-1}(\mathbf{w}^{T}\mathbf{x}+b)\tag{3.15}
$$
​		其中$g(·)$称为“联系函数”，为1时就是最简单的一元线性模型，$g(·)=ln(·)$时就是对数线性回归。

### 3.3 对数几率回归

​		对于分类任务，只需找到一个单调可微函数将分类任务的真是标记$y$与线性回归模型的预测值联系起来。

​		考虑二分类任务， 其输出标记$y\in\{0,1\}$，而线性回归模型产生的预测值$z = ω^{T}x +b$ 是实值，需将实值$z$ 转换为0/ 1 值. 最理想的是"单位阶跃函数" 
$$
y=
\begin{cases}
0, &z<0;\\
0.5, &z=0;\\
1,&z>0,
\end{cases}
\tag{3.16}
$$
![](https://img-blog.csdnimg.cn/2021071914450382.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L091dXVf,size_16,color_FFFFFF,t_70#pic_center)

​		但是单位阶跃函数==不连续==，不能直接用作$g^{-}(·)$，因此需要找到一个可以近似替代且单调可微函数-->对数几率函数：
$$
y=\frac{1}{1+e^{-z}}\tag{3.17}
$$
​		代入线性模型可得：
$$
y=\frac{1}{1+e^{-(\mathbf{w}^{T}\mathbf{x}+b)}}\tag{3.18}
$$
整理可得：
$$
ln\frac{y}{1-y}=\mathbf{w}^{T}\mathbf{x}+b\tag{3.9}
$$
​		若将$y$视为样本$\mathbf{x}$作为正例的可能性，则$1-y$则是反例可能性，两者的比值
$$
\frac{y}{1-y}\tag{3.20}
$$
称为“几率”，取对数则得到“对数几率”
$$
ln\frac{y}{1-y}\tag{3.21}
$$
​		-->实际上式（3.18）是在用线性回归模型预测结果去逼近真实标记的对数几率。

​		虽然名字是回归，但其实这是一种分类学习方法，具有较多优点：

> 1. 无需事先假设数据分布，直接对分类可能性建模；
> 2. 得到近似概率预测，对需要利用概率辅助决策很有用；
> 3. 对率函数是任意阶可导的凸函数，有很好的数学性质，现有的许多数值优化算法都可直接用于求取最优解.

​		将$y$视为类后验概率$p(y=1|\mathbf{x})$，则
$$
ln\frac{p(y=1|\mathbf{x})}{p(y=0|\mathbf{x})}=\mathbf{w}^{T}\mathbf{x}+b\tag{3.22}
$$
​		显然有
$$
p(y=1|\mathbf{x})=\frac{e^{w^{T}x+b}}{1+e^{w^{T}x+b}}\tag{3.23}
$$

$$
p(y=0|\mathbf{x})=\frac{1}{1+e^{w^{T}x+b}}\tag{3.24}
$$

​		-->可以通过极大似然估计来估计$\mathbf{\omega},b$。
$$
\ell(\omega,b)=\sum_{i=1}^{m}lnp(y_{i}|\mathbf{x}_{i};\omega,b)\tag{3.25}
$$
​		令$\beta=(\omega;b)$，$\hat{x}=(x;1)$，则$\omega^{T}x+b$可简化为$\beta^{T}\hat{x}$。再令$p_{1}(\hat{x};\beta)=p(y=1|\hat{x;\beta}),p_{0}(\hat{x};\beta)=p(y=0|\hat{x;\beta})=1-p_{1}(\hat{x};\beta)$，则（3.25）中的似然项可写为
$$
p(y_{i}|x_{i};\omega,x)=y_{i}p_{1}(\hat{x_{i}};\beta)+(1-y_{i})p_{0}(\hat{x_{i}};\beta)\tag{3.26}
$$
​		

​		将（3.26）代入（3.25），并根据（3.23）和（3.24）可知，最大化（3.25）等价于最小化
$$
\ell(\beta)=\sum_{i=1}^{m}(-y_{i}\beta^{T}\hat{x_{i}}+ln(1+e^{\beta^{T}\hat{x}}))\tag{3.27}
$$
​		式(3.27)是关于$β$ 的高阶可导连续凸函数，根据凸优化理论，经典的数值优化算法如梯度下降法、牛顿法等都可求得其最优解，于是就得到
$$
\beta^{*}=argmin_{\beta}\ell(\beta)\tag{3.28}
$$
### 3.4 线性判别分析

​		思想：给定训练样例集，设法将样例投影到一条直线上，使得同类样例的投影点尽可能接近、异类样例的投影点尽可能远离；在对新样本进行分类时，将其投影到同样的直线上，再根据投影点位置来确定新的样本的类别。

![](https://img-blog.csdnimg.cn/202107191444281.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L091dXVf,size_16,color_FFFFFF,t_70#pic_center)

​		给定数据集$D = { (x_{i},y_{i})}_{i=1}^{m},y_{i} \in\{0,1\}$， 令$x_{i},\mu,\sum_{i}$分别表示第$i\in\{0,1\}$ 类示例的集合、均值向量、协方差矩阵.若将数据投影到直线$w$上，则两类样本的中心在直线上的投影分别为$w^{T}\mu_{0}$和$w^{T}\mu_{1}$; 若将所有样本点都投影到直线上，则两类样本的协方差分别为$w^{T}\sum_{0}w$ 和$w^{T}\sum_{1}w$。==由于直线是一维空间，因此投影和协方差均为实数==。

> 同类样例投影点尽可能接近：协方差尽可能小；
>
> 异类样例投影点尽可能远离：类中心之间的距离尽可能大；即$||w^{T}\mu_{0}-w^{T}\mu_{1}||_{2}^{2}$尽可能大。

​		两者结合，则可得到欲最大化的目标
$$
\begin{aligned}
J&=\frac{||w^{T}\mu_{0}-w^{T}\mu_{1}||_{2}^{2}}{w^{T}\sum_{0}w+w^{T}\sum_{1}w}\\
&=\frac{w^{T}(\mu_{0}-\mu_{1})(\mu_{0}-\mu_{1})^{T}w}{w^{T}(\sum_{0}+\sum_{1})w}
\end{aligned}
\tag{3.32}
$$
​		定义"类内散度矩阵"：
$$
\begin{aligned}
S_{w}&=\sum\ _{0}+\sum\ _{1}\\
&=\sum_{x\in X_{0}}(x-\mu_{0})(x-\mu_{0})^{T}+\sum_{x\in X_{1}}(x-\mu_{1})(x-\mu_{1})^{T}\\
\end{aligned}
\tag{3.33}
$$
以及“类间散度矩阵”：
$$
\begin{aligned}
S_{b}&=(\mu _{0}-\mu _{1})(\mu _{0}-\mu _{1})^{T}\\
\end{aligned}
\tag{3.33}
$$
则（3.32）可重写为
$$
J=\frac{w^{T}S_{b}w}{w^{T}S_{w}w}\tag{3.35}
$$
这就是LDA的最大化目标。

​		由于（3.35）分子分母都是关于$w$的二次项，所以解与$w$的长度无关，只与其方向有关。令$w^{T}S_{w}w=1$，则（3.35）等价于
$$
\begin{aligned}
&min_{w}\ &-w^{T}S_{b}w\\
&s.t.\ &w^{T}S_{w}w\\
\end{aligned}
\tag{3.36}
$$
由拉格朗日乘子法，上式等价于
$$
S_{b}w=\lambda S_{w}w\tag{3.37}
$$
其中$\lambda$是拉格朗日乘子。注意到$S_{b}w$的方向恒为$\mu_{0}-\mu_{1}$，不妨令
$$
S_{b}w=\lambda(\mu_{0}-\mu_{1})\tag{3.38}
$$
代入（3.37）可得
$$
w=S_{w}^{-1}(\mu_{0}-\mu_{1})\tag{3.39}
$$
​		LDA可从贝叶斯决策理论的角度来阐释，并可证明当类数据同先验、满足高斯分布且协方差相等时，LDA可达到最有分类。

![](https://img-blog.csdnimg.cn/20210719144731528.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L091dXVf,size_16,color_FFFFFF,t_70#pic_center)
$$
max_{W}\frac{tr(W^TS_{b}W)}{trW^{T}S_{w}W}\tag{3.44}
$$
​		其中，$W\in \mathbb{R}^{d\times(N-1)},tr(·)$表示矩阵的迹。（3.44）可以通过如下广义特征值问题求解：
$$
S_{b}W=\lambda S_{w}W\tag{3.45}
$$
​		W的闭式解是$S_{W}^{-1}S_{b}$的N-1个最大广义特征值所对应的特征向量组成的举证。

​		若将W 视为一个投影矩阵，则多分类LDA 将样本投影到N-1 维空间，N-1 通常远小子数据原有的属性数.于是，可通过这个投影来减小样本点的维数，且投影过程中使用了类别信息，因此==LDA也常被视为一种经典的监督降维技术==。

