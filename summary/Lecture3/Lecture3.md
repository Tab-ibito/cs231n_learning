# Lecture3

## 梯度下降 gradient descent

在SVM中我们分析凸函数做优化，实际情况会比简单凸函数更复杂，但我们依然使用凸函数分析。

凸函数可能未必处处可导，因此涉及到次导数 / 次梯度概念。

### 优化方法对比

* 随机权重矩阵W：优化效率低，计算开销大
* 随机W，调整系数：略有提升，开销依然大
* 跟随梯度

### 计算梯度

* 数值方法计算：简单易实现，效率低（维数高的情况下求偏导计算量过大）
* 解析方法计算：**直接对Loss函数求偏导**

$$L_i = \sum_{j \neq y_i} \max(0, w_j^T x_i - w_{y_i}^T x_i + \Delta)$$

max函数对 `w_y_i` 求偏导，如果它非正那 = 0求偏导仍等于0，如果是正的，那么值为它本身乘以 x_i，求偏导为 x_i.

相当于

$$\nabla_{w_{y_i}} L_i = - \left( \sum_{j \neq y_i} \mathbb{1}(w_j^T x_i - w_{y_i}^T x_i + \Delta > 0) \right) x_i$$

换言之 `gradient wrt w_yi = - count * x_i`，以及 `gradient wrt w_j = x_i`，结果上依然得到一个大的矩阵。

从而在实际计算的时候减去这个负数，将Loss Function向更小的方向推。

```python
while True:
  data_batch = sample_training_data(data, 256) # sample 256 examples
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # perform parameter update
```

### 学习率

决定执行函数改变的**步长**，学习率过大可能会导致精度下降。

### batch size 和样本量

* Batch Gradient Descent：每次用全部训练样本计算梯度，方向稳定，但每步成本高
* Stochastic Gradient Descent（SGD）：每次用一个样本估计梯度，更新频繁但噪声大
* Mini-batch SGD：每次用一小批样本估计梯度，是深度学习中最常用的做法，许多语境下的 SGD 指的是 Mini-batch SGD

batch size一般设置为2的整数次幂，有利于硬件优化。

对于mini-batch中的多个样本，通常把每个样本的loss / 梯度**取平均**，得到这一步的更新方向。

### SGD过程中遇到的问题与优化算法

* 处理学习曲线导数**陡峭振荡**的问题，以及在平缓处推进缓慢的问题
* 处理**鞍点**等引发的问题
* 过程可能会被**小噪声**干扰

解决：

* 给予一个“动量”（Momentum）/ 惯性以迭代， `vx = rho * vx + dx`，`rho` 可设置为0.9或0.99。
* RMSProp等方法（后面章节会详细介绍）
