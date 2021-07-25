### 第5章 神经网络

#### 5.1神经元模型

​		神经网络是由具有适应性的简单单元（神经元）组成的广泛并行互连的网络，它的组织能够模拟生物神经系统对真实世界物体所作出的交互反应。

​		神经元接收到来自$n$​个其他神经元传递过来的输入信号，这些输入信号通过带权重的连接（connection）进行传递，神经元接收到的总输入值将与神经元的阀值进行比较，然后通过"激活函数"处理以产生神经元的输出。激活函数常用Sigmoid函数。

#### 5.2感知机与多层网络

​		感知机由两层神经元组成。

![](https://img-blog.csdnimg.cn/09956f6c1a5c415fbdd01d167c3d43f8.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L091dXVf,size_16,color_FFFFFF,t_70#pic_center)

​		能够容易地实现逻辑与、或、非运算。注意到$y=f(\sum_{i}w_{i}x_i -\theta)$，假定是阶跃函数
$$
y=
\begin{cases}
1,\quad x\geq0\\
0,\quad x<0
\end{cases}
$$
有

* “与”：令$w_1=w_2=1,\theta=2$，则$y=f(1 \cdot x_1+1 \cdot x_2-2)$，仅在$x_1=x_2=1,y=1$；
* “或”：令$w_1=w_2=1,\theta=0.5$​​​​，则$y=f(1 \cdot x_1+1 \cdot x_2-0.5)$​​​​，仅在$x_1 =1\ ||\  x_2=1,y=1$​​​​；

* “非”：令$w_1=-0.6,w_2=0,\theta=-0.5$​​​​​，则$y=f(-0.6 \cdot x_1+0 \cdot x_2+0.5)$​​​​​，在$x_1 =1,y=1;x_1=0,y=1$​.​​​​​

  ​		更一般地，给定训练集，权重和阈值都可以通过学习得到。阈值可以看作一个固定输入为-1.0的“哑结点”，所对应权重$w_{n+1}$，这样权重和阈值的学习就可以统一为权重学习。
  $$
  w_i\leftarrow w_i+ \Delta w_i\tag{5.1}
  $$

  $$
  \Delta w_i=\eta(y-\hat{y})x_i\tag{5.2}
  $$

  $\hat{y}$为当前感知机的输出，$\eta\in(0,1)$称为学习率。若感知机对训练样例预测准确，即$\hat{y}=y$，则感知机不发生变化，否则将根据错位的成都进行权重调整。

  ​		==感知机学习能力有限，只能解决线性可分问题==，否则学习过程会发生震荡，难以稳定，不能求得合适解。

  ![](https://img-blog.csdnimg.cn/13ec969de5d647ffb36dc5bdf4e6b0ce.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L091dXVf,size_16,color_FFFFFF,t_70#pic_center)

  

* 隐层：隐含层，输出层和输入层之间的一层神经元；
* 多层前馈神经网络：每层神经元与下一层神经元==全互连==，神经元之间不存在同层连接、跨层连接；
  * 输入层神经网络仅是接受输入，不进行函数处理，隐层和输出层包含功能神经元；

#### 5.3 误差拟传播算法

​		误差拟传播算法（**BP算法**）不仅可用于多层前馈神经网络，还可用于其他类型的神经网络，是迄今最成功的神经网络学习算法。

![](https://img-blog.csdnimg.cn/92b77c24ac8e4b29b95e10937ff7f30c.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L091dXVf,size_16,color_FFFFFF,t_70#pic_center)

​		给定训练集$D=\{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\},x_i\in \Re^{d},y_i\in \Re^{l}$，即输入示例由$d$个属性描述，输出$l$维实值向量。其中，输出层第$j$个神经元阈值用$\theta_{j}$表示，隐层第$h$个神经元的阈值用$\gamma_h$表示，输入层第$i$个神经元与隐层第$h$个神经元的连接权为$v_{ih}$，隐层第$h$个神经元与输出层第$j$个神经元的连接权为$w_{hj}$，输入输出如图所示，$b_h$为隐层第$h$​​个神经元的输出。假设使用Sigmoid函数。

​		对训练例$(x_k,y_k)$，假定神经网络的输出为$\hat{y}_k=(\hat{y}_1^{k},\hat{y}_2^{k},...,\hat{y}_l^{k})$，即
$$
\hat{y}_{j}^{k}=f(\beta_j-\theta_j)\tag{5.3}
$$
则网络在训练例上的均方误差为
$$
E_k=\frac{1}{2}\sum_{j=1}^{l}(\hat{y}_{j}^{k}-y_{j}^{k})^{2}\tag{5.4}
$$
​		图5.7中有$(d+l+1)q+l$个参数需要确定：输入层到隐层有$d*q$个权值，隐层到输出层有$q*l$个权值、$q$个隐层神经元阈值、$l$个输出层神经元阈值。BP算法是一个迭代算法，迭代的每一轮采用广义的感知机学习规则对参数进行更新估计，即与（5.1）类似，参数更新估计式
$$
v\leftarrow v+\Delta v\tag{5.5}
$$
​		学习率控制算法迭代的更新步长。

​		BP算法的目标是要最小化训练集$D$上的累计误差
$$
E=\frac{1}{m}\sum_{k=1}^{m}E_k\tag{5.16}
$$
​		基于累计误差最小化的更新规则就得到了累计误差逆转播算法，累积BP和标准BP都很常用：

> 标准BP算法每次更新只针对单个样例，参数更新频繁，对不同样例进行更新的效果可能出现“抵消现象”，迭代次数多；
>
> 累积BP算法直接针对累计误差最小化，读取整个训练集再对参数更新，频率低但是速度缓慢；

防止过拟合：

* 早停：将数据分为训练集和验证集合，训练集用来计算梯度、更新连接权和阔值，验证集用来估计误差，若训练集误差降低但验证集误差升高，则停止训练，同时返回具有最小验证集误差的连接权和阑值.

* 正则化：基本思想是在误差目标函数中增加一个用于描述网络复杂度的部分，例如连接权和阈值的平方和，令$E_k$表示第$k$个训练样例上的误差，$w_i$表示连接权和阈值，则误差目标函数改变为
  $$
  E=\lambda \frac{1}{m}\sum_{k=1}^{m}E_k+(1-\lambda)\sum_i w_i^2 \tag{5.17}
  $$
  其中$\lambda\in(0,1)$​用于对经验误差与网络复杂度这两项进行折中，常通过交叉验证法来估计.



#### 5.4全局最小和局部极小

​		对于$\omega^*$和$\theta^*$​，若存在$\varepsilon>0$使得
$$
\forall (\omega;\theta)\in\{(\omega;\theta)|\ ||(\omega;\theta)-(\omega^*;\theta^*)||\ \leq \varepsilon \}
$$
都有$E(\omega;\theta)\geq E(\omega^*;\theta^*)$，成立，则$(\omega^*;\theta^*)$为局部最小解；若对参数空间任意$(\omega;\theta)$都有$E(\omega;\theta)\geq E(\omega^*;\theta^*)$，则$(\omega^*;\theta^*)$​为全局最小解。

#### 5.5其他常见神经网络

* RBF网络：径向基函数网络，单隐层前馈神经网络；
  $$
  \varphi(x)=\sum_{i=1}^{q}w_i\rho(x,c_i)\tag{5.18}
  $$

* ART网络：自适应谐振理论网络

  * 无监督学习策略
  * “胜者通吃”

* SOM网络：自组织映射网络

  * 竞争学习型无监督神经网络；

* 及联相关网络

* Elman网络：最常用的递归神经网络之一

* Boltzmann机：基于能量的模型

