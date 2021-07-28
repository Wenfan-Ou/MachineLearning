### 第6章 支持向量机

#### 6.1 间隔与支持向量

​		在分类学习中找到能将训练样本分开的划超平面，要求该平面的“容忍性”最好，即所产生的分类结果是最鲁棒的，对未见示例的泛化能力最强。

​		在样本空间中，划分超平面可通过如下线性方程来描述：
$$
w^Tx+b=0\tag{6.1}
$$
其中$w=(w_1;w_2;...;w_d)$为法向量，决定了超平面的方向；$b$为位移项，决定了超平面与原点之间的距离。样本空间中任意点$x$到超平面$(w,b)$的距离可写为
$$
r=\frac{|w^Tx+b|}{||w||}\tag{6.2}
$$
​		假设超平面可以正确分类，即$(x_i,y_i)\in D$，若$y_i=+1$，则有$w^Tx_i+b>0$；若$y_i=-1$，则有$w^Tx_i+b<0$。令
$$
\begin{cases}
&w^Tx_i+b\geq+1, &y_i=+1;\\
&w^Tx_i+b\leq+1, &y_i=-1;\\
\end{cases}
\tag{6.3}
$$
![](https://img-blog.csdnimg.cn/9b9cd8416d9d483eb8e424c0c810aecc.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L091dXVf,size_16,color_FFFFFF,t_70#pic_center)

​		如图所示，距离超平面最近的这几个训练样本点使（6.3）成立的等号成立，它们被称为“支持向量”，两个异类支持向量到超平面的距离之和为
$$
\gamma = \frac{2}{||w||}\tag{6.4}
$$
它被称为“间隔”。

​		想找到具有最大间隔的划分超平面，也就是让（6.3）参数使得$\gamma$最大，即
$$
\begin{aligned}
&max_{w,b}\quad  \frac{2}{||w||}\\
&s.t. \quad \quad\  y_i(w^Tx_i+b)\geq1,\ i=1,2,..,m
\end{aligned}
\tag{6.5}
$$
显然，（6.5）等价于
$$
\begin{aligned}
&min_{w,b}\quad  \frac{1}{2}||w||^2\\
&s.t. \quad\ y_i(w^Tx_i+b)\geq1,\ i=1,2,..,m
\end{aligned}
\tag{6.6}
$$
这就是支持向量机（SVM）的基本型。

#### 6.2对偶问题

目的：求解（6.6）得到大间隔划分超平面所对应的模型
$$
f(x)=w^Tx+b\tag{6.7}
$$
注意到（6.6）本身是一个凸二次规划问题，可以用优化计算包求解。

**更高效的方法**:

​		对于（6.6）使用拉格朗日乘子法可得到其“对偶问题”。该问题的拉格朗日函数可写为
$$
L(w,b,\alpha)=\frac{1}{2}||w||^2+\sum_{i+1}^{m}\alpha_{i}(1-y_i(w^Tx_i+b))\tag{6.8}
$$
其中$\alpha=(\alpha_1;\alpha_2;...;\alpha_m)$。令$L(w,b,\alpha)$对$w,b$求偏导为0可得
$$
w=\sum_{i=1}^{m}\alpha_iy_ix_i\tag{6.9}
$$

$$
0=\sum_{i=1}^{m}\alpha_iy_i\tag{6.10}
$$

代入拉格朗日函数（6.8），再考虑（6.10）的约束，就得到了（6.6）的对偶问题
$$
max_{\alpha}\sum_{i=1}^{m}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}y_iy_jx_i^Tx_j\tag{6.11}
$$

$$
\begin{aligned}
s.t.\ &\sum_{i=1}^{m}\alpha_iy_i=0,\\
&\alpha_i \geq0,\quad i=1,2,...,m
\end{aligned}
$$

解出$\alpha$，求出$w,b$即可得到模型
$$
\begin{aligned}
f(x)&=w^Tx+b\\
&=\sum_{i=1}^{m}\alpha_iy_ix_i^Tx+b
\end{aligned}
\tag{6.12}
$$
​		从对偶问题（6.11）解出的问是式（6.8） 中的拉格朗日乘子，它恰对应着训
练样本$(x_i,y_i)$.  注意到式（6.6） 中有不等式约束，因此上述过程需满足KKT
条件，即要求
$$
\begin{cases}
&\alpha_i\geq0;\\
&y_if(x_i)-1\geq0;\\
&a_i(y_if(x_i)-1)=0;
\end{cases}
\tag{6.13}
$$
所以

* 若$\alpha_i=0$，则样本不会在（6.12）的求和中出现，不产生影响；

* 若$\alpha_i>0$，则必有$y_if(x_i)=1$，所对应的样本点位于最大间隔边界上，是一个支持向量；

* 支持向量机训练完成后，大部分样本都不需要保留，最终模型仅与支持向量有关。

  ​	（6.11）作为一个二次规划问题，可以用SMO算法求解。

> SMO 的基本思路是先固定$\alpha_i$之外的所有参数，然后求$\alpha_i$上的极值.由于
> 存在约束$\sum_{i=1}^m \alpha_iy_i=0$ ，若固定$\alpha_i$之外的其他变量，则$\alpha_i$ 可由其他变量导出.
> 于是， SMO 每次选择两个变量$\alpha_i$和$\alpha_j$​，并固定其他参数.这样，在参数初始化后， SMO 不断执行如下两个步骤直至收敛:
>
> * 选取一队需要更新的变量$\alpha_i$和$\alpha_j$；
> * 固定$\alpha_i$和$\alpha_j$以外的参数，求解（6.11）获得更新后的$\alpha_i$和$\alpha_j$。

​		注意到只需选取的$\alpha_i$和$\alpha_j$​中有一个不满足KKT 条件(6.13) ，目标函数就
会在选代后减小. 直观来看， KKT 条件违背的程度越大，则变量更新后可能导致的目标函数值减幅越大.于是， SMO 先选取违背KKT 条件程度最大的变量.第二个变量应选择一个使目标函数值减小最快的变量，但由于比较各变量所对应的目标函数值减幅的==复杂度过高==，因此SMO 采用了一个启发式：==使选取的两变量所对应样本之间的问隔最大==。一种直观的解释是，这样的两个变量有很大的差别，与对两个相似的变量进行更新相比，对它们进行更新会带给目标函数值更大的变化.

#### 6.3核函数

​		对于非线性可分的问题，可将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间内线性可分。例如将原始的二维空间映射到一个合

![](https://img-blog.csdnimg.cn/a1669bf8620a4b5281eaea785790feed.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L091dXVf,size_16,color_FFFFFF,t_70#pic_center)

适的三维空间，就能找到一个合适的划分超平面。若属性数有限，那么一定存在一个高维特征空间使样本可分。

​		令$\phi(x)$表示$x$映射后的向量特征，若求解$\phi(x_i)^T\phi(x_j)$是困难的，可以设想一个“核函数”
$$
k(x_i,x_j)=<\phi(x_i),\phi(x_j)>=\phi(x_i)^T\phi(x_j)\tag{6.22}
$$
即$x_i,x_j$在特征空间的内积等于它们在原始样本空间中通过$k(·，·)$计算的结果。求解后得到
$$
f(x)=\sum_{i=1}^{m}\alpha_iy_ik(x,x_i)+b\tag{6.24}
$$
显示出模型最优解可通过训练样本的核函数展开，这一展式亦称为“支持向量展式”。

**定理6.1（核函数）**

![](https://img-blog.csdnimg.cn/d96bc84515b14026b2cba53be5f44732.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L091dXVf,size_16,color_FFFFFF,t_70#pic_center)

![](https://img-blog.csdnimg.cn/58f4b6f806d442b9929af3b8a5848dbe.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L091dXVf,size_16,color_FFFFFF,t_70#pic_center)

![](https://img-blog.csdnimg.cn/8f5adecef59a4734a0b2bbf918a1f6da.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L091dXVf,size_16,color_FFFFFF,t_70#pic_center)

#### 6.4软间隔和正则化

* 现实任务中很难确定合适的核函数使得训练样本在特征空间中线性可分；
* 就算找到了难以断定可分结果是否由过拟合造成；
* 缓解的方法是==允许支持向量机在一些样本上出错==

![](https://img-blog.csdnimg.cn/ecf168c958ba449dae6b370c57cb0dbb.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L091dXVf,size_16,color_FFFFFF,t_70#pic_center)

​		具体来说，“硬间隔”要求所有样本均满足约束（6.3）
$$
\begin{cases}
&w^Tx_i+b\geq+1, &y_i=+1;\\
&w^Tx_i+b\leq+1, &y_i=-1;\\
\end{cases}
\tag{6.3}
$$
而软间隔允许某些样本不满足约束
$$
y_i(w^Tx_i+b)\ge1\tag{6.28}
$$
在最大化间隔时，不满足约束的样本应尽可能少，于是优化目标可写为
$$
min_{w,b}\frac{1}{2}||w||^2+C\sum_{i=1}^{m}\ell_{0/1}(y_i(w^Tx_i+b)-1)\tag{6.29}
$$
其中$C>0$是一个常数，$\ell_{0/1}$是“0/1损失函数”
$$
\ell_{0/1}(z)=
\begin{cases}
&1,&if z<0;\\
&0,&otherwise.
\end{cases}
\tag{6.30}
$$

* 当$C\rightarrow \infty$，（6.29）迫使所有样本满足（6.28），（6.29）等价（6.6）；
* 当$C$取有限值，（6.29）允许一些样本不满足约束。

由于$\ell_{0/1}$​数学性质不好，导致（6.29）不易求解，选用一些函数替代，称为“替代损失”
$$
hinge损失：\ell_{hinge}(z)=max(0,1-z)\tag{6.31}
$$

$$
指数损失：\ell_{exp}(z)=exp(-z)\tag{6.32}
$$

$$
对率损失：\ell_{log}(z)=log(1+exp(-z))\tag{6.33}
$$

![](https://img-blog.csdnimg.cn/9e37a8fb726c4d6a9225fe57c4b5f481.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L091dXVf,size_16,color_FFFFFF,t_70#pic_center)

​		我们还可以把（6.29）中的0/1损失函数换成别的替代损失函数以得到其他学习模型，这些模型的性质与所用的替代函数直接相关，但它们具有一个共性：优化目标中的第一项用来描述划分超平面的“间隔”大小，另一项$\sum_{i=1}^m\ell(f(x_i),y_i)$用来表述训练集上的误差，可写为更一般的形式
$$
min_f\Omega(f)+C\sum_{i=1}^{m}\ell(f(x_i),y_i)\tag{6.42}
$$
其中$\Omega(f)$称为“结构风险”，用来描述$f$的某些性质，第二项称为“经验风险”，用于描述模型与训练数据的契合程度；C用于对二者折中。

![](https://img-blog.csdnimg.cn/f79a7f530d5543688b334dbf6602ae65.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L091dXVf,size_16,color_FFFFFF,t_70#pic_center)

#### 6.5支持向量回归

对于样本$(x,y)$

* 传统回归模型：基于模型输出$f(x)$与真实输出$y$之间的差别来计算损失，当$f(x)=y$，损失才为零；
* 支持向量回归：假设能容忍$f(x)$与$y$之间最多有$\epsilon$的偏差，即$|f(x)-y|>\epsilon$才开始计算损失。​

![](https://img-blog.csdnimg.cn/be0063dd4e294fef83755b2c50efbdf9.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L091dXVf,size_16,color_FFFFFF,t_70#pic_center)

​		SVR问题可形式化为
$$
min_{w,b}\frac{1}{2}||w||^2+C\sum_{i=1}^{m}\ell_{\epsilon}(f(x_i)-y_i)\tag{6.43}
$$
C是正则化常数，$\ell_{\epsilon}$是$\epsilon$​-不敏感损失函数
$$
\ell_{\epsilon}(z)=
\begin{cases}
&0,&if|z|\leq \epsilon\\
&|z|-\epsilon,&otherwise
\end{cases}
\tag{6.44}
$$
引入松弛变量$\xi_i$和$\hat{\xi_i}$
$$
min_{w,b,\xi_i,\hat{\xi_i}}\frac{1}{2}||w||^2+C\sum_{i=1}^m(\xi_i+\hat{\xi_i})\tag{6.45}
$$

$$
\begin{aligned}
s.t.\ &f(x_i)-y_i\le\epsilon+\xi_i,\\
&y_i-f(x_i)\le \epsilon+\hat{\xi_i},\\
&\xi_i\ge0,\hat{\xi_i}\ge0,i=1,2,...,m
\end{aligned}
$$

类似的，通过引入拉格朗日乘子得到拉格朗日函数，对四个参数求偏导为零，代入（6.46）得到SVR的对偶问题，通过要求满足KKT条件得到解形如
$$
f(x)=\sum_{i=1}^{m}(\hat{\alpha_i}-\alpha_i)x_i^Tx+b\tag{6.53}
$$


