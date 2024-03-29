1. [TOC]

[注意]：部分公式显示不完整，建议下载到本地查看

# 演化计算

## 遗传算法

### 定义

英语：genetic algorithm（GA）,是计算数学中用于解决最优化的搜索方法，是进化算法的一种。进化算法最初是借鉴了进化生物学中的一些现象而发展起来了，这些现象包括**遗传**，**变异**，**自然选择**以及**杂交**等。

### 算法步骤

1. 初始化种群
2. 循环
   1. 评价种群中的个体适应度
   2. 按照比例选择下一个种群（轮盘法，竞争法）
   3. 种群个体之间杂交以及变异
3. 直到满足一定条件（精度或者迭代次数）



### 相关参数

- 种群规模（Population size）：即种群中染色体的个体数量

- 字符串长度(string length)：染色体的长度

- 交配概率(probability of performing crossover)： 控制着交配算子的使用频率。交配操作可以加快收敛，使解达到最有希望的最佳解区域，因此一般取较大的交配概率，但交配概率太高也可能导致过早收敛，则称为早熟。 

  >  这部分直接复制的还不是很理解 

- 变异概率（probability of mutation）：控制个体的变异概率

### 例题以及求解代码

- 选择策略-转盘式选择

$$
p_i=\frac{f_i}{\Sigma{f_i}}
$$

​		<img src="http://latex.codecogs.com/gif.latex?p_i=\frac{f_i}{\Sigma{f_i}}" title="p_i=\frac{f_i}{\Sigma{f_i}}" />

其中$p_i$代表第$i$个体被选择的概率，$f_i$代表第$i$个体的适应度，也就是一般问题中需要求解的函数值。

- 杂交策略-均匀杂交

  随机选择当前种群中的另外一个个体，同时选择一个与个体一样长的二进制串作为模板，在模板中，如果某位是1那么两个父体在此位交换，如果是0 ，则不处理。

- 变异

  按照变异概率$p_m$将某些分量在定义域内随机取值。

#### 题目

$$
min\prod^n_{i=1}\sum^5_{j=1}jcos[(j+1)x_i+j]
$$

<img src="http://latex.codecogs.com/gif.latex?min\prod^n_{i=1}\sum^5_{j=1}jcos[(j&plus;1)x_i&plus;j]" title="min\prod^n_{i=1}\sum^5_{j=1}jcos[(j+1)x_i+j]" />

其中 $-10\leq{x_i}\le10,i=1,2,...,n$，当n=1、 2、 3和4时分别有3、 18、 81和324 个不同的全局最优解。做n=1的情况，至少要得到一个全局最优解，可用十进制编码也可用二进制编码。用二进制编码时至少精确到小数点后2位，用十进制编码时至少精确到小数点后8位。  

见代码文件 evo_1.py,evo_1_n=2.py

由于在原来的代码上进行了一部分修改，以使其适应n>1的情况。原来的参考链接找不到，如果你找到了这个代码对应的参考文献，请联系我，我会及时补上。

- 算法结果部分n=1截图

  ![](1574307883111.png)

- n=2截图

  ![](1574307923972.png)

- n=3截图

  ![](1574307933912.png)

- n=4截图

  ![](1574308015035.png)

## 粒子群算法

### 定义

 		粒子群算法的思想源于对鸟/鱼群捕食行为的研究，模拟鸟集群飞行觅食的行为，鸟之间通过集体的协作使群体达到最优目的，是一种基于Swarm Intelligence的优化方法。它没有遗传算法的“交叉”(Crossover) 和“变异”(Mutation) 操作，它通过追随当前搜索到的最优值来寻找全局最优。粒子群算法与其他现代优化方法相比的一个明显特色就是所**需要调整的参数很少、简单易行**，收敛速度快，已成为现代优化方法领域研究的热点。 

### 算法介绍

​		每个寻优的问题解都被想像成一只鸟，称为“粒子”。所有粒子都在一个D维空间进行搜索。所有的粒子都由一个适应值函数确定适应值以判断目前的位置好坏。每一个粒子必须赋予记忆功能，能记住所搜寻到的最佳位置。每一个粒子还有一个速度以决定飞行的距离和方向。这个速度根据它本身的飞行经验以及同伴的飞行经验进行动态调整。

### 算法步骤

![](1574307837145.png)

### 例题以及求解代码

同上

代码见PSO_1_modify.py

- 结果截图

  ![](1574308081642.png)

## 参考文献

1.  [https://zh.wikipedia.org/wiki/%E9%81%97%E4%BC%A0%E7%AE%97%E6%B3%95](https://zh.wikipedia.org/wiki/遗传算法) 访问时间：2019年11月21日
2.  吴志健老师的PPT
3.  https://www.cnblogs.com/21207-iHome/p/6062535.html  访问时间：2019年11月21日
4.  遗传算法对应的参考链接暂时找不到了