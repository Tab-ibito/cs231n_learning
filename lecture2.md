# CS231n Lecture 2 总结：Image Classification with Linear Classifiers

## 1. 本讲主题

Lecture 2 继续讨论图像分类问题，并从最基础的“数据驱动方法”一路过渡到线性分类器与 Softmax loss，为后续神经网络和卷积神经网络做铺垫。

核心内容包括：

- 图像分类任务的困难点
- 数据驱动方法
- k-Nearest Neighbor, kNN
- 线性分类器
- 从代数、视觉、几何角度理解线性分类
- Softmax loss

## 2. 图像分类问题

图像分类的目标是：

> 输入一张图像，输出它所属的类别标签。

例如输入一张猫的图片，模型需要判断它是 `cat`，而不是 `dog`、`car` 或 `airplane`。

对人类来说这很自然，但对计算机来说，图像只是一个三维数字数组：

```text
height x width x channels
```

例如 CIFAR-10 中一张图片大小是：

```text
32 x 32 x 3 = 3072
```

每个像素值通常在 `[0, 255]` 之间。

## 3. Semantic Gap：语义鸿沟

图像分类的核心困难在于：

> 人类看到的是“猫”“车”“飞机”等语义对象，而计算机看到的是像素矩阵。

这就是 semantic gap。

模型必须从低层像素值中学习高层语义概念。

常见挑战包括：

- **Viewpoint variation**：同一个物体从不同角度看，像素会完全不同。
- **Illumination**：光照变化会显著改变像素值。
- **Deformation**：物体形状可能发生变化，比如猫蜷缩、伸展。
- **Occlusion**：物体可能被遮挡。
- **Background clutter**：背景可能和目标物体混在一起。
- **Intra-class variation**：同一类别内部差异很大，比如不同品种的猫差别明显。

## 4. 数据驱动方法

早期计算机视觉试图手写规则，比如：

```python
def classify_image(image):
    # 手工检测边缘、角点、形状
    return label
```

但这种方法很难扩展，因为我们无法手写出所有类别的视觉规则。

CS231n 强调使用数据驱动方法：

1. 收集带标签的数据集。
2. 用训练数据学习分类器。
3. 在测试数据上评估模型表现。

图像分类 pipeline 可以概括为：

```text
Input: labeled training images
Learning: train a classifier
Evaluation: test on unseen images
```

## 5. k-Nearest Neighbor 分类器

kNN 是最简单的数据驱动分类方法之一。

它的思想是：

> 对一张测试图片，在训练集中找到最相似的 k 张图片，然后用它们的标签投票。

当 `k = 1` 时，就是 nearest neighbor classifier。

### 距离度量

常见距离包括：

#### L1 distance

```text
d(I1, I2) = sum(|I1 - I2|)
```

也就是逐像素相减，取绝对值后求和。

#### L2 distance

```text
d(I1, I2) = sqrt(sum((I1 - I2)^2))
```

也就是欧氏距离。

### kNN 的问题

kNN 虽然直观，但有明显缺点：

- 训练阶段几乎只是记住数据。
- 测试阶段很慢，因为要和所有训练样本比较。
- 对高维图像数据效果有限。
- 像素距离不一定等价于语义相似。

所以 kNN 更适合作为入门基线，而不是实际深度视觉系统的核心方法。

## 6. 线性分类器

为了克服 kNN 测试慢的问题，课程引入 parametric model。

线性分类器不再记住所有训练图片，而是学习一组参数：

```text
f(x, W, b) = Wx + b
```

其中：

- `x` 是展平后的图像向量
- `W` 是权重矩阵
- `b` 是偏置
- 输出是每个类别的 score

以 CIFAR-10 为例：

```text
x: 3072 x 1
W: 10 x 3072
b: 10 x 1
scores: 10 x 1
```

每个类别都会得到一个分数，分数最高的类别就是模型预测结果。

## 7. 线性分类器的三种理解方式

### 7.1 代数视角

线性分类器就是矩阵乘法：

```text
scores = Wx + b
```

每一行 `W[j]` 可以看成第 `j` 类的分类模板，与输入图像做点积后得到该类分数。

### 7.2 视觉视角

如果把 `W` 的某一行重新 reshape 成图片形状，就可以把它看成某个类别的“模板”。

例如：

- airplane 模板可能偏好蓝色背景
- horse 模板可能捕捉某种轮廓
- car 模板可能偏好特定形状和颜色

但线性分类器只能为每一类学到一个模板，因此难以处理同一类别的多种姿态、颜色和背景变化。

### 7.3 几何视角

每张图片都可以看成高维空间中的一个点。

线性分类器在这个空间中学习一些超平面，用来把不同类别分开。

问题是：真实图像类别往往不是线性可分的，所以线性模型能力有限。

## 8. Loss Function：如何衡量模型好坏

分类器输出 scores 后，需要一个 loss function 判断这些 scores 是否好。

直觉上：

> 正确类别的分数应该高于错误类别的分数。

loss 越小，说明模型越符合训练数据。

本讲重点讨论 Softmax loss。

## 9. Softmax Classifier

Softmax classifier 仍然使用线性 score function：

```text
f(x, W) = Wx
```

但它会把原始分数解释为 unnormalized log probabilities。

Softmax 函数将 scores 转换成概率分布：

```text
P(y = j | x) = exp(s_j) / sum_k exp(s_k)
```

然后使用 cross-entropy loss：

```text
L_i = -log(P(correct class | x_i))
```

如果正确类别概率接近 1，loss 接近 0。

如果正确类别概率很低，loss 会很大。

## 10. Softmax 的直觉

Softmax 的目标是：

> 让正确类别的概率尽可能高，让错误类别的概率尽可能低。

它和 SVM loss 的区别是：

- SVM 关心正确类别分数是否比错误类别高出 margin。
- Softmax 关心正确类别的概率是否足够大。

Softmax 输出更容易解释为“置信度”，但这些概率并不一定是真正校准过的概率。

## 11. 本讲重点总结

Lecture 2 的主线是：

```text
Image Classification
-> Data-driven approach
-> kNN
-> Linear classifier
-> Score function
-> Loss function
-> Softmax loss
```

最重要的思想是：

- 图像分类不能靠手写规则解决。
- 模型需要从数据中学习。
- kNN 是最简单的数据驱动方法，但测试慢、效果有限。
- 线性分类器用参数 `W` 和 `b` 学习从图像到类别分数的映射。
- Softmax loss 用概率和交叉熵来衡量分类结果。
- 后续神经网络可以看作在线性分类器基础上的扩展。

## 12. 一句话理解

CS231n Lecture 2 的核心是：

> 从“直接比较图片”的 kNN，过渡到“学习参数并输出类别分数”的线性分类器，再用 Softmax loss 定义模型该如何变好。

## References

- [Stanford CS231N Spring 2025 Lecture 2: Image Classification with Linear Classifiers](https://www.youtube.com/watch?v=pdqofxJeBN8)
- [CS231n Notes: Image Classification](https://cs231n.github.io/classification/)
- [CS231n Notes: Linear Classification](https://cs231n.github.io/linear-classify/)
