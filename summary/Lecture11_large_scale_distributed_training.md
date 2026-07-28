# Lecture 11

## Communications Collectives

在大型的多个gpu上计算，如何调度数据。

* All-Reduce: 不同板子上数据加和，并把所有数据分发给各个板子
* Reduce-Scatters: 将数据分块切片，每一块分给一个板子
* All-Gather: Yi = concat(X1, …, XN)，把所有板子切块拼起来，再把所有数据分发给各个板子
* All-to-All: 将数据块重排分配

## How to train on lots of GPUs

Transformer model activations have shape (Layer, Batch, Sequence, Channel)

### Data Parallelism

Use **minibatch** of MN samples, **split** over M GPUs.

Loss由各个GPU的Batch均值计算得到；可以梯度回滚给每个gpu。

各个GPU的梯度相加取均值，分别更新每个GPU的权值矩阵。

*问题：Model size constrained by GPU memory.*

一个权值需要 `weight` `grad` `beta_1 (Adam)` `beta_2 (Adam)` 四个参数，每个GPU都要维护一套权重矩阵，占用空间大。

解决：Fully Sharded Data Parallelism (FSPD) 

* **Split** model weights across GPUs
* 每个权重矩阵只存在一个GPU里面
* 前向传播时在每个GPU计算 `layer[i-1]` 时把 `layer[i]` 的矩阵广播给所有GPU
* 计算完成后把其他GPU的这一层**复制品删掉**
* **反向传播**时也做类似回滚，每个GPU把各自算的梯度投给这一层矩阵的Owner。
* Optimization: don’t delete last weight at end of forward to avoid immediately resending it.

注意这里每一个GPU拿到不同的Batch，意味着输入不一样。

#### Activation Checkpointing

已有方法的问题：神经层的Activations保存在GPU内存里面。

解决：不将所有的Activations保存，而在每一次反向传播时顺带重新计算。  
本质是**时间换空间**的处理，用 `O(N^2)` 的计算时间换取 `O(1)` 的存储空间。

进一步权衡的方法：分段设置Checkpoint，保存相应Activations。

#### GPU性能发挥指标

* Hardware FLOPs Utilization (HFU)
  * 我们达到了GPU理论上矩阵计算性能的多少比率
    * 在计算矩阵乘法的时候是最佳情况，大矩阵乘法的对应值更高
  * 问题：不考虑激活检查点或“辅助”计算
* Model FLOPs Utilization (MFU)
  * GPU理论峰值浮点运算能力中，有多少比例被用于“有效”的模型计算？
    * 理论和实际矩阵浮点乘法用时的比值
    *  \>30% is good, \>40% is excellent

### Context Parallelism

Transformer输入线性长度序列，使用多个GPU分析**同一个序列的不同维度**。

N路情况下，每个GPU负责 `(batch, sequence/N, channels)` 的张量。

* 便于归一化、残差连接，进行并行计算（没有权重影响）
* MLP层面要考虑**权重**影响。
  * Each GPU keeps a copy of the weights and communicates gradients like in DP.
* Attention & Attention operator的运行机制更复杂。
  * (Option 1) Ulysses: 每个GPU维护attention heads，将QKV计算后 re-shard from `(batch, sequence / N, heads, head dim)` to `(batch, sequence, heads / N, head dim)`
    * Easy to implement, but **heads must be divisible** by GPUs
  * (Option 2) Ring Attention: 分块分配给各个GPU。分配循环内层对应 key & value，外层对应 queries
    * Complex to implement but can scale to very long sequences
* QKV Projection 类似于DP机制。

### Pipeline Parallelism

把训练的不同层分配给不同GPU。

问题：GPU间有**序列依赖性**。

解决：Run multiple **microbatches** at the same time, pipeline them through the GPUs.

### Tensor Parallelism

把整个权重矩阵分开（按照行或者列切分），每一块相应的矩阵乘法交给对应的GPU

问题：需要把结果拼接起来，需要时序和GPU间的Communications

解决：用**两层TP层**，一层按列分一层按行分。
* XW = Y (layer 1)
* YU = Z (layer 2)