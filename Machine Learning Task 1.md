## Machine Learning Task 1：第1章 & 第2章

### 绪论

1. 基本术语

   ​	假定收集了一批数据记录，记录的集合就是**数据集**；每一条记录（对事件或者对象的描述）则是一个**示例**；反应事件或者对象的某种特殊性质的是**属性** || **特征**；属性可取的值则为**属性值**；属性值张成的空间则为**属性空间**；在一个属性空间里，每个事件或者对象都有自己对应的坐标，因此也将示例成为**特征向量**。

   ​	但仅仅通过数据集的内容并不能做出判断，要建立关于==<u>“预测”</u>==的模型。

   ​	例如：

   ​			（（色泽=青绿，根蒂=蜷缩，敲声=浊响），好瓜）

   ​	这条记录里的“好瓜”就是一种标记（label），拥有了标记的示例就是样例（example）。

   ​	一般地，用
   $$
   (x_{i},y{i})\quad y_{i}\in Y
   $$
   表示第$$i$$个样例，$$y_{i}$$是$$x_i$$的标记，$$Y$$是所有标记的集合，也被称为“标记空间”。

   ​		学得模型后，使用其进行预测的过程称为“测试”，被测试的样本被称为“测试样本”。例如在学得$$f$$后，对于测试例$$x$$，可得其预测标记$y=f(x)$.

​		学习任务分类：

> 监督学习（supervised learning）：代表有分类和回归
>
> ​	分类：预测值为离散值，例如“好”，“坏”；
>
> > ​	二分类：正类和负类；
> >
> > ​	多分类：多个分类类别；
>
> > 回归：预测值为连续值，例如成熟度；
>
> 无监督学习（unsupervised learning）：代表有聚类
>
> ​	聚类：数据内具有潜在概念的示例形成多个“簇”（cluster），通常不含有标记信息

2. 假设空间

   ​	学习过程可视为一个在所有<u>假设</u>（学得模型对应了关于数据的某种潜在规律）组成的空间中进行搜索的过程，搜索目标就是找到与训练集匹配的假设，即能够将训练集中对事件或者对象判断正确的假设。假设的表示一旦确定，假设空间及其规模大小就确定了。

   ​	现实的假设空间很大，但是学习过程是基于有限样本学习的，因此可能有多个假设与训练集一致（或者理解为通过训练集学得的假设有多个），我们称为“版本空间”。

   ![ML1_1.png](https://github.com/Wenfan-Ou/MachineLearning/raw/main/ML1_1.png)

3. 归纳偏好：机器学习算法在学习过程中对某种类型假设的偏好

> 根据“奥卡姆剃刀”原则，更平滑、更简单的是“正确”的归纳偏好 

结论推导：训练集外的误差为
$$
E_{ote}(\tau_{a}|X,f )=\sum_{h}\sum_{x\in \chi-X}P(x)\amalg(h(x)\neq f(x))P(h|X,\tau_a)
$$
其中，$\amalg(·)$是指示函数，·为真取1，否则取0

> $\tau_{a}$表示$a$算法；
>
> $P(h|X,\tau_{a})$为算法在训练数据$X$上产生假设$h$的概率；
>
> $f$为真实函数；

考虑二分问题，且真实目标函数可以是任何函数$\chi\rightarrow\{0,1\}$，函数空间为$\{0,1\}^{|\chi|}$。对所有可能的$f$按均匀分布对误差求和，有
$$
\begin{aligned}
\sum_{f}E_{ote}(\tau_{a}|X,f )&=\sum_{f}\sum_{h}\sum_{x\in \chi-X}P(x)\amalg(h(x)\neq f(x))P(h|X,\tau_a)\\
&=\sum_{x\in \chi-X}P(x)\sum_{h}P(h|X,\tau_a)\sum_{f}\amalg(h(x)\neq f(x))\\
&=\sum_{x\in \chi-X}P(x)\sum_{h}P(h|X,\tau_a)\frac{1}{2}2^{|\chi|}\\
&=2^{|\chi|-1}\sum_{x\in \chi-X}P(x)·1\\
&=2^{|\chi|-1}\sum_{x\in \chi-X}P(x)\\
\end{aligned}
$$
==step 2->step 3==:

​	$f$是任何一个可以映射到$\chi$的函数，每个$f$对应到$\chi$的每一个对象都有可能产生两种结果，因此最后将各个对象的可能结果组合起来就是$2^{\chi}$种可能，因此$\sum_{f}\amalg(h(x)\neq f(x))=2^{\chi}$

​	==因此，总误差与算法无关！==

 	NFL定理最重要的寓意在于，若考虑所有潜在问题，则所有的学习算法都一样好/糟糕，因此学习算法自身的归纳偏好与问题是否匹配，往往起到决定性作用.

### 模型评估与选择

​	学习器的预测输出与样本的真实输出之间的差异被称为“误差”，在训练集上的误差被称为“训练误差”或者“经验误差”；在新样本上的误差被称为“泛化误差”。

​	显然，我们希望我们的学习器泛化误差能尽可能小，这样它的泛化能力（普适性）更好。我们并不能预先知道泛化误差，只能知道经验误差，但经验误差过小甚至为0并不好——学习器把训练样本学得“太好了”的时候，很可能已经把训练样本自身的一些特点当作了所有潜在样本都会具有的一般特质，从而导致泛化能力下降，这就是==“过拟合”==。与之相对的==“欠拟合”==就是样本的一般特质学习不够。

​	为了选择最适合的模型，缓解“过拟合”，要进行模型评估。

1. 评估方法

   ​	只有一个包含$m$个样例的数据集$D=\{(x_{1},y_{1}),...,(x_{m},y_{m})\}$，既要训练也要测试，因此需要对$D$进行适当的处理，常见方法有：

> 留出法：划分两个相斥的集合，$D=S\cup T,S\cap T =\varnothing$.
>
> ​	-->要注意的是划分集合时，集合内数据的概率分布一致性要保持统一，不要引入额外的差异；
>
> ​	-->通常来说2/3~4/5的样本用于训练，剩下的用来测试；
>
> 交叉验证法：划分$k$个相斥集合，每次挑选一个作为测试集，其他用于训练；将$k$个精度结果求均值
>
> ​	-->常取$k=10$；
>
> ​	-->要注意的是划分集合时，集合内数据的概率分布一致性要保持统一，不要引入额外的差异；
>
> 自助法：bootstrap sampling，对数据集$D$采样m次形成新数据集$D'$
>
> ​	-->样本在$m$次采样不被采到的概率为
> $$
> \lim_{m\rightarrow\infty}(1-\frac{1}{m})^{m}=\frac{1}{e}\approx0.368
> $$
> ​		可将$D'$作为训练集，D \ D'作为测试集；
>
> ​	-->实际评估的模型与期望评估的模型都使用m个样本，且仍有36.8%的样本未在训练集中出现；
>
> ​	-->在数据集小、难以有效划分的时候很有用；

2. 调参与最终模型：基于验证集进行调参
3. 性能度量

> 错误率与精度
>
> 查准率、查全率与F 1
>
> ROC与AUC
>
> > ROC——受试者工作特征
> >
> > -->ROC的纵轴是“真正例率”，横轴是“假正例率”，分别定义为
> > $$
> > TPR = \frac{TP}{TP + FN}\\
> > FPR = \frac{FP}{FP + TN}\\
> > $$
> > ![ML1_2.png](https://github.com/Wenfan-Ou/MachineLearning/raw/main/ML1_2.png)
> >
> > AUC——ROC曲线下的面积
>
> 代价敏感错误率与代价曲线

4. 比较检验

   > 假设检验
   >
   > 交叉验证t检验
   >
   > McNemer检验
   >
   > Friedman检验和Nemenyi检验
   >
   > 偏差与方差
