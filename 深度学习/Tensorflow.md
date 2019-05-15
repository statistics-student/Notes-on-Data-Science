# ${Tensorflow}$

##${Tensorflow}$编程基础

| 名词           | 解释    |
| :------------- | :---------------------------- |
| 张量           | 数据，即某一类型的多维数组    |
| 变量           | 一般定义模型参数              |
| 占位符         | 输入变量的载体                |
| OP操作（节点） | 一个OP获得0个或多个tensorflow |

## 函数

###**tf.matmul**：

矩阵计算

###tf.constant()

定义常量

### tf.assign(state, new_value)

将 State 变量更新成 new_value

**该函数定义一个变量**

```python
hello=tf.constant('hello,tensorflow!')
sess=tf.Session()
sess.run(hello)
sess.close()
```

##实例1

```python
import tensorflow as tf
import numpy as np

#create data

x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

#搭建模型
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))
y=Weights*x_data+biases

#计算误差
loss=tf.reduce_mean(tf.square(y-y_data))#最小二乘损失

#传播误差
optimizer=tf.train.GradientDescentOptimizer(0.5)#optimizer:优化器;GrGradientDescentOptimizer(0.5)梯度下降
train=optimizer.minimize(loss)

#初始化
init=tf.global_variables_initializer()#初始化所有的变量

#训练
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20==0:
            print(step,sess.run(Weights),sess.run(biases))
```

## 实例2：Placeholder 传入值

placeholder 是 Tensorflow 中的占位符，暂时储存变量. 

Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(), 然后以这种形式传输数据

```python
sess.run(*, feed_dict={input: }).
```

```python
#在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul = multiply 是将input1和input2 做乘法运算，并输出为 output 
ouput = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))
# [ 14.]
```

## 实例3：定义add_layer

```python
#定义四个参数
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)#biases不推荐全为0
    Wx_plus_b=tf.matmul(inputs,Weights)+biases#神经网络未激活的值
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs
```

## 实例4：构建神经网络

```python
#导入数据
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]#np.newaxis在此处加一维
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

#定义隐藏层
L1=add_layer(xs,1,10,activation_function=tf.nn.relu)

#输出层
prediction=add_layer(L1,10,1,activation_function=None)

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer() 

sess = tf.Session()
sess.run(init)

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
    # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))`
```

##补充（session）

**交互式session**：

```python
sess=tf.InteractiveSession()
```

**Supervisor方式session**:

可以自动管理session的具体任务

## 指定GPU

```python
with tf.Session() as sess:
	with tf.device("/gpu:1"):
		a = tf.placeholder(tf.int16)
		b = tf.placeholder(tf.int16)
		add = tf.add(a, b)
		……
```

**目前支持的设备**

```python
·cpu：0：机器的CPU。
·gpu：0：机器的第一个GPU，如果有的
话。
·gpu：1：机器的第二个GPU，依此类推。
```

**tf.ConfigProto**

自己在config中构建使用gpu

+ log_device_placement=True：是否打印设备
  分配日志。
+ allow_soft_placement=True：如果指定的设
  备不存在，允许TF自动分配设备。

**example**

```python
config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
session = tf.Session(config=config, ...)
```

## 设置GPU使用资源

```python
config.gpu_options.allow_growth = True#按需分配
```

```python
gpu_options = tf.GPUOptions(allow_growth=True)
config=tf.ConfigProto(gpu_options=gpu_options)#也可在config创建时指定
```

```python
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7代表分配给tensorflow的GPU显存大小为：GPU实际显存×0.7。
```

## 保存模型和载入模型

### 保存模型

首先建立一个saver()

```python
#之前是各种构建模型graph的操作(矩阵相乘，sigmoid等)
saver = tf.train.Saver() #生成saver
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer()) #先对模型初始化
	#然后将数据丢入模型进行训练blablabla
	#训练完以后，使用saver.save 来保存
	saver.save(sess, "save_path/file_name")
	#file_name如果不存在，会自动创建
```

### 载入模型

```python
saver = tf.train.Saver()
with tf.Session() as sess:
	#参数可以进行初始化，也可不进行初始化。即使初始化了，初始化的值也会被restore的值给覆盖
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, "save_path/file_name") #会将已经保存的变
```

## 实例5：保存模型和载入模型

```python
tf.reset_default_graph()#防止出错
#create data

x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

#搭建模型
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
y=Weights*X+biases

#计算误差
loss=tf.reduce_mean(tf.square(y-Y))#最小二乘损失

#传播误差
optimizer=tf.train.GradientDescentOptimizer(0.5)#optimizer:优化器;GrGradientDescentOptimizer(0.5)梯度下降
train=optimizer.minimize(loss)

#初始化
init=tf.global_variables_initializer()#初始化所有的变量
#保存模型

saver=tf.train.Saver()
#训练
'''
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train,feed_dict={X:x_data,Y:y_data})
        if step%20==0:
            print(step,sess.run(Weights),sess.run(biases))
    saver.save(sess,r'E:/Tensorflow/linefunction.cpkt')#保存模型
'''
with tf.Session()as sess2:
    sess2.run(init)
    saver.restore(sess2,'E:\\Tensorflow\\linefunction.cpkt')#载入模型
    print("x=0.2，y=", sess2.run(y,feed_dict={X:0.2})[0])
```

###打印保存模型内容

```python
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file('E:\\Tensorflow\\linefunction.cpkt',tensor_name='sess2',all_tensors=True)
```

### 保存模型的其他方法 

```python
saver = tf.train.Saver({'weight': W, 'bias': b})
''''''''''''''''''''''''''''''''''''''''''''''''
saver = tf.train.Saver([W, b])
tf.train.Saver({v.op.name: v for v in [W, b]})
```

### 保存检查点和载入检查点

```python
saver = tf.train.Saver(max_to_keep=1) #max_to_keep=1，表明最多只保存一个检查点文件。
saver.save(sess, savedir+"linermodel.cpkt", global_step=epoch)#在保存时使用了如下代码传入了迭代次数
'''
example:
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", epoch+1, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))
            if not (loss == "NA" ):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
            saver.save(sess, savedir+"linermodel.cpkt", global_step=epoch)
'''
#再次载入时需要这样写
saver.restore(sess2, savedir+"linermodel.cpkt-" + str(load_epoch))
```

### 更简便的保存检查点<font color=red>--------tf.train.MonitoredTraining Session </font>

```python
import tensorflow as tf
tf.reset_default_graph()
global_step = tf.train.get_or_create_global_step()
step = tf.assign_add(global_step, 1)

with tf.train.MonitoredTrainingSession(checkpoint_dir='log/checkpoints',save_checkpoint_secs  = 2) as sess:#save_checkpoint_secs = 2表示每训练2秒保存一次检查点
    print(sess.run([global_step]))
    while not sess.should_stop():
        i = sess.run( step)
        print( i)
```

##模型操作常用函数

## Tensorboard可视化

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


plotdata = { "batchsize":[], "loss":[] }
def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


#生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
#图形显示
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()


tf.reset_default_graph()

# 创建模型
# 占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")
# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# 前向结构
z = tf.multiply(X, W)+ b
tf.summary.histogram('z',z)#将预测值以直方图显示
#反向优化
cost =tf.reduce_mean( tf.square(Y - z))
tf.summary.scalar('loss_function', cost)#将损失以标量显示
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent

# 初始化变量
init = tf.global_variables_initializer()
#参数设置
training_epochs = 20
display_step = 2

# 启动session
with tf.Session() as sess:
    sess.run(init)
    
    merged_summary_op = tf.summary.merge_all()#合并所有summary
    #创建summary_writer，用于写文件
    summary_writer = tf.summary.FileWriter('log/mnist_with_summaries',sess.graph)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
            
            #生成summary
            summary_str = sess.run(merged_summary_op,feed_dict={X: x, Y: y});
            summary_writer.add_summary(summary_str, epoch);#将summary 写入文件

        #显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", epoch+1, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))
            if not (loss == "NA" ):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)

    print (" Finished!")
    print ("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))
    #print ("cost:",cost.eval({X: train_X, Y: train_Y}))

    #图形显示
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
     
    plt.show()

    print ("x=0.2，z=", sess.run(z, feed_dict={X: 0.2}))
   
```

###打开Tensorboard

```python
tensorboard --logdir E://Tensorflow//tensorboard-exercise1//mnist_with_summaries
```

##Tensorflow常用函数

###tf函数

| 操作组                     | 操作                                                 |
| -------------------------- | ---------------------------------------------------- |
| Maths                      | Add, Sub, Mul, Div, Exp, Log, Greater, Less, Equal   |
| Array                      | Concat, Slice, Split, Constant, Rank, Shape, Shuffle |
| Matrix                     | MatMul, MatrixInverse, MatrixDeterminant             |
| Neuronal Network           | SoftMax, Sigmoid, ReLU, Convolution2D, MaxPool       |
| Checkpointing              | Save, Restore                                        |
| Queues and syncronizations | Enqueue, Dequeue, MutexAcquire, MutexRelease         |
| Flow control               | Merge, Switch, Enter, Leave, NextIteration           |

### TensorFlow的算术操作如下

| 操作                        | 描述                                                         |
| --------------------------- | ------------------------------------------------------------ |
| tf.add(x, y, name=None)     | 求和                                                         |
| tf.sub(x, y, name=None)     | 减法                                                         |
| tf.mul(x, y, name=None)     | 乘法                                                         |
| tf.div(x, y, name=None)     | 除法                                                         |
| tf.mod(x, y, name=None)     | 取模                                                         |
| tf.abs(x, name=None)        | 求绝对值                                                     |
| tf.neg(x, name=None)        | 取负 (y = -x).                                               |
| tf.sign(x, name=None)       | 返回符号 y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0. |
| tf.inv(x, name=None)        | 取反                                                         |
| tf.square(x, name=None)     | 计算平方 (y = x * x = x^2).                                  |
| tf.round(x, name=None)      | 舍入最接近的整数 # ‘a’ is [0.9, 2.5, 2.3, -4.4] tf.round(a) ==> [ 1.0, 3.0, 2.0, -4.0 ] |
| tf.sqrt(x, name=None)       | 开根号 (y = \sqrt{x} = x^{1/2}).                             |
| tf.pow(x, y, name=None)     | 幂次方  # tensor ‘x’ is [[2, 2], [3, 3]] # tensor ‘y’ is [[8, 16], [2, 3]] tf.pow(x, y) ==> [[256, 65536], [9, 27]] |
| tf.exp(x, name=None)        | 计算e的次方                                                  |
| tf.log(x, name=None)        | 计算log，一个输入计算e的ln，两输入以第二输入为底             |
| tf.maximum(x, y, name=None) | 返回最大值 (x > y ? x : y)                                   |
| tf.minimum(x, y, name=None) | 返回最小值 (x < y ? x : y)                                   |
| tf.cos(x, name=None)        | 三角函数cosine                                               |
| tf.sin(x, name=None)        | 三角函数sine                                                 |
| tf.tan(x, name=None)        | 三角函数tan                                                  |
| tf.atan(x, name=None)       | 三角函数ctan                                                 |

###张量操作Tensor Transformations

+ 数据类型转换Casting

  | 操作                                                         | 描述                                                         |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | tf.string_to_number (string_tensor, out_type=None, name=None) | 字符串转为数字                                               |
  | tf.to_double(x, name=’ToDouble’)                             | 转为64位浮点类型–float64                                     |
  | tf.to_float(x, name=’ToFloat’)                               | 转为32位浮点类型–float32                                     |
  | tf.to_int32(x, name=’ToInt32’)                               | 转为32位整型–int32                                           |
  | tf.to_int64(x, name=’ToInt64’)                               | 转为64位整型–int64                                           |
  | tf.cast(x, dtype, name=None)                                 | 将x或者x.values转换为dtype # tensor `a` is [1.8, 2.2], dtype=tf.float tf.cast(a, tf.int32) ==> [1, 2] # dtype=tf.int32 |

+ 形状操作Shapes and Shaping

  | 操作                                  | 描述                                                         |
  | ------------------------------------- | ------------------------------------------------------------ |
  | tf.shape(input, name=None)            | 返回数据的shape # ‘t’ is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]] shape(t) ==> [2, 2, 3] |
  | tf.size(input, name=None)             | 返回数据的元素数量 # ‘t’ is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]] size(t) ==> 12 |
  | tf.rank(input, name=None)             | 返回tensor的rank 注意：此rank不同于矩阵的rank， tensor的rank表示一个tensor需要的索引数目来唯一表示任何一个元素 也就是通常所说的 “order”, “degree”或”ndims” #’t’ is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]] # shape of tensor ‘t’ is [2, 2, 3] rank(t) ==> 3 |
  | tf.reshape(tensor, shape, name=None)  | 改变tensor的形状 # tensor ‘t’ is [1, 2, 3, 4, 5, 6, 7, 8, 9] # tensor ‘t’ has shape [9] reshape(t, [3, 3]) ==>  [[1, 2, 3], [4, 5, 6], [7, 8, 9]] #如果shape有元素[-1],表示在该维度打平至一维 # -1 将自动推导得为 9: reshape(t, [2, -1]) ==>  [[1, 1, 1, 2, 2, 2, 3, 3, 3], [4, 4, 4, 5, 5, 5, 6, 6, 6]] |
  | tf.expand_dims(input, dim, name=None) | 插入维度1进入一个tensor中 #该操作要求-1-input.dims() # ‘t’ is a tensor of shape [2] shape(expand_dims(t, 0)) ==> [1, 2] shape(expand_dims(t, 1)) ==> [2, 1] shape(expand_dims(t, -1)) ==> [2, 1] <= dim <= input.dims() |

  - 切片与合并（Slicing and Joining）

  | 操作                                                         | 描述                                                         |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | tf.slice(input_, begin, size, name=None)                     | 对tensor进行切片操作 其中size[i] = input.dim_size(i) - begin[i] 该操作要求 0 <= begin[i] <= begin[i] + size[i] <= Di for i in [0, n] #’input’ is  #[[[1, 1, 1], [2, 2, 2]],[[3, 3, 3], [4, 4, 4]],[[5, 5, 5], [6, 6, 6]]] tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]] tf.slice(input, [1, 0, 0], [1, 2, 3]) ==>  [[[3, 3, 3], [4, 4, 4]]] tf.slice(input, [1, 0, 0], [2, 1, 3]) ==>  [[[3, 3, 3]], [[5, 5, 5]]] |
  | tf.split(split_dim, num_split, value, name=’split’)          | 沿着某一维度将tensor分离为num_split tensors # ‘value’ is a tensor with shape [5, 30] # Split ‘value’ into 3 tensors along dimension 1 split0, split1, split2 = tf.split(1, 3, value) tf.shape(split0) ==> [5, 10] |
  | tf.concat(concat_dim, values, name=’concat’)                 | 沿着某一维度连结tensor t1 = [[1, 2, 3], [4, 5, 6]] t2 = [[7, 8, 9], [10, 11, 12]] tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]] tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]] 如果想沿着tensor一新轴连结打包,那么可以： tf.concat(axis, [tf.expand_dims(t, axis) for t in tensors]) 等同于tf.pack(tensors, axis=axis) |
  | tf.pack(values, axis=0, name=’pack’)                         | 将一系列rank-R的tensor打包为一个rank-(R+1)的tensor # ‘x’ is [1, 4], ‘y’ is [2, 5], ‘z’ is [3, 6] pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # 沿着第一维pack pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]] 等价于tf.pack([x, y, z]) = np.asarray([x, y, z]) |
  | tf.reverse(tensor, dims, name=None)                          | 沿着某维度进行序列反转 其中dim为列表，元素为bool型，size等于rank(tensor) # tensor ‘t’ is  [[[[ 0, 1, 2, 3], #[ 4, 5, 6, 7],  #[ 8, 9, 10, 11]], #[[12, 13, 14, 15], #[16, 17, 18, 19], #[20, 21, 22, 23]]]] # tensor ‘t’ shape is [1, 2, 3, 4] # ‘dims’ is [False, False, False, True] reverse(t, dims) ==> [[[[ 3, 2, 1, 0], [ 7, 6, 5, 4], [ 11, 10, 9, 8]], [[15, 14, 13, 12], [19, 18, 17, 16], [23, 22, 21, 20]]]] |
  | tf.transpose(a, perm=None, name=’transpose’)                 | 调换tensor的维度顺序 按照列表perm的维度排列调换tensor顺序， 如为定义，则perm为(n-1…0) # ‘x’ is [[1 2 3],[4 5 6]] tf.transpose(x) ==> [[1 4], [2 5],[3 6]] # Equivalently tf.transpose(x, perm=[1, 0]) ==> [[1 4],[2 5], [3 6]] |
  | tf.gather(params, indices, validate_indices=None, name=None) | 合并索引indices所指示params中的切片 ![tf.gather](http://img.blog.csdn.net/20160808174705034) |
  | tf.one_hot (indices, depth, on_value=None, off_value=None,  axis=None, dtype=None, name=None) | indices = [0, 2, -1, 1] depth = 3 on_value = 5.0  off_value = 0.0  axis = -1  #Then output is [4 x 3]:  output =  [5.0 0.0 0.0] // one_hot(0)  [0.0 0.0 5.0] // one_hot(2)  [0.0 0.0 0.0] // one_hot(-1)  [0.0 5.0 0.0] // one_hot(1) |

###矩阵相关运算

| 操作                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| tf.diag(diagonal, name=None)                                 | 返回一个给定对角值的对角tensor # ‘diagonal’ is [1, 2, 3, 4] tf.diag(diagonal) ==>  [[1, 0, 0, 0] [0, 2, 0, 0] [0, 0, 3, 0] [0, 0, 0, 4]] |
| tf.diag_part(input, name=None)                               | 功能与上面相反                                               |
| tf.trace(x, name=None)                                       | 求一个2维tensor足迹，即对角值diagonal之和                    |
| tf.transpose(a, perm=None, name=’transpose’)                 | 调换tensor的维度顺序 按照列表perm的维度排列调换tensor顺序， 如为定义，则perm为(n-1…0) # ‘x’ is [[1 2 3],[4 5 6]] tf.transpose(x) ==> [[1 4], [2 5],[3 6]] # Equivalently tf.transpose(x, perm=[1, 0]) ==> [[1 4],[2 5], [3 6]] |
| tf.matmul(a, b, transpose_a=False,  transpose_b=False, a_is_sparse=False,  b_is_sparse=False, name=None) | 矩阵相乘                                                     |
| tf.matrix_determinant(input, name=None)                      | 返回方阵的行列式                                             |
| tf.matrix_inverse(input, adjoint=None, name=None)            | 求方阵的逆矩阵，adjoint为True时，计算输入共轭矩阵的逆矩阵    |
| tf.cholesky(input, name=None)                                | 对输入方阵cholesky分解， 即把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解A=LL^T |
| tf.matrix_solve(matrix, rhs, adjoint=None, name=None)        | 求解tf.matrix_solve(matrix, rhs, adjoint=None, name=None) matrix为方阵shape为[M,M],rhs的shape为[M,K]，output为[M,K] |

### 复数操作

| 操作                                                | 描述                                                         |
| --------------------------------------------------- | ------------------------------------------------------------ |
| tf.complex(real, imag, name=None)                   | 将两实数转换为复数形式 # tensor ‘real’ is [2.25, 3.25] # tensor `imag` is [4.75, 5.75] tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]] |
| tf.complex_abs(x, name=None)                        | 计算复数的绝对值，即长度。 # tensor ‘x’ is [[-2.25 + 4.75j], [-3.25 + 5.75j]] tf.complex_abs(x) ==> [5.25594902, 6.60492229] |
| tf.conj(input, name=None)                           | 计算共轭复数                                                 |
| tf.imag(input, name=None) tf.real(input, name=None) | 提取复数的虚部和实部                                         |
| tf.fft(input, name=None)                            | 计算一维的离散傅里叶变换，输入数据类型为complex64            |

### 归约计算(Reduction)

| 操作                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| tf.reduce_sum(input_tensor, reduction_indices=None,  keep_dims=False, name=None) | 计算输入tensor元素的和，或者安照reduction_indices指定的轴进行求和 # ‘x’ is [[1, 1, 1] # [1, 1, 1]] tf.reduce_sum(x) ==> 6 tf.reduce_sum(x, 0) ==> [2, 2, 2] tf.reduce_sum(x, 1) ==> [3, 3] tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]] tf.reduce_sum(x, [0, 1]) ==> 6 |
| tf.reduce_prod(input_tensor,  reduction_indices=None,  keep_dims=False, name=None) | 计算输入tensor元素的乘积，或者安照reduction_indices指定的轴进行求乘积 |
| tf.reduce_min(input_tensor,  reduction_indices=None,  keep_dims=False, name=None) | 求tensor中最小值                                             |
| tf.reduce_max(input_tensor,  reduction_indices=None,  keep_dims=False, name=None) | 求tensor中最大值                                             |
| tf.reduce_mean(input_tensor,  reduction_indices=None,  keep_dims=False, name=None) | 求tensor中平均值                                             |
| tf.reduce_all(input_tensor,  reduction_indices=None,  keep_dims=False, name=None) | 对tensor中各个元素求逻辑’与’ # ‘x’ is  # [[True, True] # [False, False]] tf.reduce_all(x) ==> False tf.reduce_all(x, 0) ==> [False, False] tf.reduce_all(x, 1) ==> [True, False] |
| tf.reduce_any(input_tensor,  reduction_indices=None,  keep_dims=False, name=None) | 对tensor中各个元素求逻辑’或’                                 |
| tf.accumulate_n(inputs, shape=None,  tensor_dtype=None, name=None) | 计算一系列tensor的和 # tensor ‘a’ is [[1, 2], [3, 4]] # tensor `b` is [[5, 0], [0, 6]] tf.accumulate_n([a, b, a]) ==> [[7, 4], [6, 14]] |
| tf.cumsum(x, axis=0, exclusive=False,  reverse=False, name=None) | 求累积和 tf.cumsum([a, b, c]) ==> [a, a + b, a + b + c] tf.cumsum([a, b, c], exclusive=True) ==> [0, a, a + b] tf.cumsum([a, b, c], reverse=True) ==> [a + b + c, b + c, c] tf.cumsum([a, b, c], exclusive=True, reverse=True) ==> [b + c, c, 0] |

### 分割

| 操作                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| tf.segment_sum(data, segment_ids, name=None)                 | 根据segment_ids的分段计算各个片段的和 其中segment_ids为一个size与data第一维相同的tensor 其中id为int型数据，最大id不大于size c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]]) tf.segment_sum(c, tf.constant([0, 0, 1])) ==>[[0 0 0 0]  [5 6 7 8]] 上面例子分为[0,1]两id,对相同id的data相应数据进行求和, 并放入结果的相应id中， 且segment_ids只升不降 |
| tf.segment_prod(data, segment_ids, name=None)                | 根据segment_ids的分段计算各个片段的积                        |
| tf.segment_min(data, segment_ids, name=None)                 | 根据segment_ids的分段计算各个片段的最小值                    |
| tf.segment_max(data, segment_ids, name=None)                 | 根据segment_ids的分段计算各个片段的最大值                    |
| tf.segment_mean(data, segment_ids, name=None)                | 根据segment_ids的分段计算各个片段的平均值                    |
| tf.unsorted_segment_sum(data, segment_ids, num_segments, name=None) | 与tf.segment_sum函数类似， 不同在于segment_ids中id顺序可以是无序的 |
| tf.sparse_segment_sum(data, indices,  segment_ids, name=None) | 输入进行稀疏分割求和 c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]]) # Select two rows, one segment. tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))  ==> [[0 0 0 0]] 对原data的indices为[0,1]位置的进行分割， 并按照segment_ids的分组进行求和 |

### 序列比较与索引提取(Sequence Comparison and Indexing)

| 操作                                   | 描述                                                         |
| -------------------------------------- | ------------------------------------------------------------ |
| tf.argmin(input, dimension, name=None) | 返回input最小值的索引index                                   |
| tf.argmax(input, dimension, name=None) | 返回input最大值的索引index                                   |
| tf.listdiff(x, y, name=None)           | 返回x，y中不同值的索引                                       |
| tf.where(input, name=None)             | 返回bool型tensor中为True的位置 # ‘input’ tensor is  #[[True, False] #[True, False]] # ‘input’ 有两个’True’,那么输出两个坐标值. # ‘input’的rank为2, 所以每个坐标为具有两个维度. where(input) ==> [[0, 0], [1, 0]] |
| tf.unique(x, name=None)                | 返回一个元组tuple(y,idx)，y为x的列表的唯一化数据列表， idx为x数据对应y元素的index # tensor ‘x’ is [1, 1, 2, 4, 4, 4, 7, 8, 8] y, idx = unique(x) y ==> [1, 2, 4, 7, 8] idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4] |
| tf.invert_permutation(x, name=None)    | 置换x数据与索引的关系 # tensor `x` is [3, 4, 0, 2, 1] invert_permutation(x) ==> [2, 4, 3, 0, 1] |

### 神经网络

+ 激活函数

  | 操作                                                         | 描述                                                         |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | tf.nn.relu(features, name=None)                              | 整流函数：max(features, 0)                                   |
  | tf.nn.relu6(features, name=None)                             | 以6为阈值的整流函数：min(max(features, 0), 6)                |
  | tf.nn.elu(features, name=None)                               | elu函数，exp(features) - 1 if < 0,否则features [Exponential Linear Units (ELUs)](http://arxiv.org/abs/1511.07289) |
  | tf.nn.softplus(features, name=None)                          | 计算softplus：log(exp(features) + 1)                         |
  | tf.nn.dropout(x, keep_prob,  noise_shape=None, seed=None, name=None) | 计算dropout，keep_prob为keep概率 noise_shape为噪声的shape    |
  | tf.nn.bias_add(value, bias, data_format=None, name=None)     | 对value加一偏置量 此函数为tf.add的特殊情况，bias仅为一维， 函数通过广播机制进行与value求和, 数据格式可以与value不同，返回为与value相同格式 |
  | tf.sigmoid(x, name=None)                                     | y = 1 / (1 + exp(-x))                                        |
  | tf.tanh(x, name=None)                                        | 双曲线切线激活函数                                           |

+ 卷积函数

  | 操作                                                         | 描述                                                         |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | tf.nn.conv2d(input, filter, strides, padding,  use_cudnn_on_gpu=None, data_format=None, name=None) | 在给定的4D input与 filter下计算2D卷积 输入shape为 [batch, height, width, in_channels] |
  | tf.nn.conv3d(input, filter, strides, padding, name=None)     | 在给定的5D input与 filter下计算3D卷积 输入shape为[batch, in_depth, in_height, in_width, in_channels] |

+ 池化函数

  | 操作                                                         | 描述                                                         |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | tf.nn.avg_pool(value, ksize, strides, padding,  data_format=’NHWC’, name=None) | 平均方式池化                                                 |
  | tf.nn.max_pool(value, ksize, strides, padding,  data_format=’NHWC’, name=None) | 最大值方法池化                                               |
  | tf.nn.max_pool_with_argmax(input, ksize, strides, padding, Targmax=None, name=None) | 返回一个二维元组(output,argmax),最大值pooling，返回最大值及其相应的索引 |
  | tf.nn.avg_pool3d(input, ksize, strides,  padding, name=None) | 3D平均值pooling                                              |
  | tf.nn.max_pool3d(input, ksize, strides,  padding, name=None) | 3D最大值pooling                                              |

+ 数据标准化

  | 操作                                                         | 描述                                                         |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)         | 对维度dim进行L2范式标准化 output = x / sqrt(max(sum(x**2), epsilon)) |
  | tf.nn.sufficient_statistics(x, axes, shift=None,  keep_dims=False, name=None) | 计算与均值和方差有关的完全统计量 返回4维元组,*元素个数，*元素总和，*元素的平方和，*shift结果 [参见算法介绍](https://www.google.com/url?q=https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data&usg=AFQjCNG5RoY7Xvpv4xg-Wy-UJvAPh2zDQw) |
  | tf.nn.normalize_moments(counts, mean_ss, variance_ss, shift, name=None) | 基于完全统计量计算均值和方差                                 |
  | tf.nn.moments(x, axes, shift=None,  name=None, keep_dims=False) | 直接计算均值与方差                                           |

+ 损失函数

  | 操作                        | 描述                     |
  | --------------------------- | ------------------------ |
  | tf.nn.l2_loss(t, name=None) | output = sum(t ** 2) / 2 |

+ 分类函数（Classification）

  | 操作                                                         | 描述                                                         |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | tf.nn.sigmoid_cross_entropy_with_logits (logits, targets, name=None)* | 计算输入logits, targets的交叉熵                              |
  | tf.nn.softmax(logits, name=None)                             | 计算softmax softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j])) |
  | tf.nn.log_softmax(logits, name=None)                         | logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))   |
  | tf.nn.softmax_cross_entropy_with_logits (logits, labels, name=None) | 计算logits和labels的softmax交叉熵 logits, labels必须为相同的shape与数据类型 |
  | tf.nn.sparse_softmax_cross_entropy_with_logits (logits, labels, name=None) | 计算logits和labels的softmax交叉熵                            |
  | tf.nn.weighted_cross_entropy_with_logits (logits, targets, pos_weight, name=None) | 与sigmoid_cross_entropy_with_logits()相似， 但给正向样本损失加了权重pos_weight |

+ 符号嵌入（Embeddings）

  | 操作                                                         | 描述                                                         |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | tf.nn.embedding_lookup (params, ids, partition_strategy=’mod’,  name=None, validate_indices=True) | 根据索引ids查询embedding列表params中的tensor值 如果len(params) > 1，id将会安照partition_strategy策略进行分割 1、如果partition_strategy为”mod”， id所分配到的位置为p = id % len(params) 比如有13个ids，分为5个位置，那么分配方案为： [[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]] 2、如果partition_strategy为”div”,那么分配方案为： [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]] |
  | tf.nn.embedding_lookup_sparse(params,  sp_ids, sp_weights, partition_strategy=’mod’,  name=None, combiner=’mean’) | 对给定的ids和权重查询embedding 1、sp_ids为一个N x M的稀疏tensor， N为batch大小，M为任意，数据类型int64 2、sp_weights的shape与sp_ids的稀疏tensor权重， 浮点类型，若为None，则权重为全’1’ |

+ 循环神经网络

  | 操作                                                         | 描述                                                         |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | tf.nn.rnn(cell, inputs, initial_state=None, dtype=None,  sequence_length=None, scope=None) | 基于RNNCell类的实例cell建立循环神经网络                      |
  | tf.nn.dynamic_rnn(cell, inputs, sequence_length=None,  initial_state=None, dtype=None, parallel_iterations=None,  swap_memory=False, time_major=False, scope=None) | 基于RNNCell类的实例cell建立动态循环神经网络 与一般rnn不同的是，该函数会根据输入动态展开 返回(outputs,state) |
  | tf.nn.state_saving_rnn(cell, inputs, state_saver, state_name,  sequence_length=None, scope=None) | 可储存调试状态的RNN网络                                      |
  | tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputs,  initial_state_fw=None, initial_state_bw=None, dtype=None, sequence_length=None, scope=None) | 双向RNN, 返回一个3元组tuple (outputs, output_state_fw, output_state_bw) |

```
— tf.nn.rnn简要介绍— 
cell: 一个RNNCell实例 
inputs: 一个shape为[batch_size, input_size]的tensor 
initial_state: 为RNN的state设定初值，可选 
sequence_length：制定输入的每一个序列的长度，size为[batch_size],值范围为[0, T)的int型数据 
其中T为输入数据序列的长度 
@ 
@针对输入batch中序列长度不同，所设置的动态计算机制 
@对于在时间t，和batch的b行，有 
(output, state)(b, t) = ? (zeros(cell.output_size), states(b, sequence_length(b) - 1)) : cell(input(b, t), state(b, t - 1))
```

+ 求值网络（Evaluation）

  | 操作                                               | 描述                                                         |
  | -------------------------------------------------- | ------------------------------------------------------------ |
  | tf.nn.top_k(input, k=1, sorted=True, name=None)    | 返回前k大的值及其对应的索引                                  |
  | tf.nn.in_top_k(predictions, targets, k, name=None) | 返回判断是否targets索引的predictions相应的值 是否在在predictions前k个位置中， 返回数据类型为bool类型，len与predictions同 |

+ [监督候选采样网络（Candidate Sampling）](https://www.tensorflow.org/versions/r0.10/extras/candidate_sampling.pdf)

  | 操作                                                         | 描述                                                         |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | **Sampled Loss Functions**                                   |                                                              |
  | tf.nn.nce_loss(weights, biases, inputs, labels, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=False, partition_strategy=’mod’, name=’nce_loss’) | 返回noise-contrastive的训练损失结果                          |
  | tf.nn.sampled_softmax_loss(weights, biases, inputs, labels,  num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=True, partition_strategy=’mod’,  name=’sampled_softmax_loss’) | 返回sampled softmax的训练损失 [参考- Jean et al., 2014第3部分](http://arxiv.org/pdf/1412.2007.pdf) |
  | **Candidate Samplers**                                       |                                                              |
  | tf.nn.uniform_candidate_sampler(true_classes, num_true,  num_sampled, unique, range_max, seed=None, name=None) | 通过均匀分布的采样集合 返回三元tuple 1、sampled_candidates 候选集合。 2、期望的true_classes个数，为浮点值 3、期望的sampled_candidates个数，为浮点值 |
  | tf.nn.log_uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None) | 通过log均匀分布的采样集合，返回三元tuple                     |
  | tf.nn.learned_unigram_candidate_sampler (true_classes, num_true, num_sampled, unique,  range_max, seed=None, name=None) | 根据在训练过程中学习到的分布状况进行采样 返回三元tuple       |
  | tf.nn.fixed_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, vocab_file=”,  distortion=1.0, num_reserved_ids=0, num_shards=1,  shard=0, unigrams=(), seed=None, name=None) | 基于所提供的基本分布进行采样                                 |

+ 保存与恢复变量

  | 操作                                                         | 描述                                                 |
  | ------------------------------------------------------------ | ---------------------------------------------------- |
  | 类tf.train.Saver(Saving and Restoring Variables)             |                                                      |
  | tf.train.Saver.__init__(var_list=None, reshape=False,  sharded=False, max_to_keep=5,  keep_checkpoint_every_n_hours=10000.0,  name=None, restore_sequentially=False, saver_def=None, builder=None) | 创建一个存储器Saver var_list定义需要存储和恢复的变量 |
  | tf.train.Saver.save(sess, save_path, global_step=None,  latest_filename=None, meta_graph_suffix=’meta’, write_meta_graph=True) | 保存变量                                             |
  | tf.train.Saver.restore(sess, save_path)                      | 恢复变量                                             |
  | tf.train.Saver.last_checkpoints                              | 列出最近未删除的checkpoint 文件名                    |
  | tf.train.Saver.set_last_checkpoints(last_checkpoints)        | 设置checkpoint文件名列表                             |
  | tf.train.Saver.set_last_checkpoints_with_time(last_checkpoints_with_time) | 设置checkpoint文件名列表和时间戳                     |

## 共享变量

### get_variable()

get_variable(<name>,<shape>,<initializer>)

指定name属性获取唯一标识

tf.Variable()

```python
var1=tf.Variable(1.0,name='firstvar')
print('var1:',var1.name)
var1 = tf.Variable(2.0 , name='firstvar')
print ("var1:",var1.name)
var2 = tf.Variable(3.0 )
print ("var2:",var2.name)
var2 = tf.Variable(4.0 )
print ("var1:",var2.name)
```

<font color='red'>指定了名字会生成两个var1,但是使用var1是第二个生效,没有指明名字的话会自动生成名字，例如var2</font>

```python
get_var1 = tf.get_variable("firstvar",[1], initializer=tf.constant_initializer(0.3))
print ("get_var1:",get_var1.name)
get_var1 = tf.get_variable("firstvar",[1], initializer=tf.constant_initializer(0.4))#崩溃
get_var1 = tf.get_variable("firstvar1",[1], initializer=tf.constant_initializer(0.4))#不会崩溃
print ("get_var1:",get_var1.name)
```

<font color='green'>使用get_variable()只能定义一次指定名称的变量，即创建两个名字相同的变量是行不通的</font>

<strong>强行使用的话</strong>(隔开作用域)

```python
with tf.variable_scope("test1", ):
    var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
    
with tf.variable_scope("test2"):
    var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
```

```python
print('var1=',var1.name)#产生一个作用域
```

**scope()**

scope()支持嵌套，如下：

```python
with tf.variable_scope("test1", ):
    var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
    
    with tf.variable_scope("test2"):
        var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
        
print ("var1:",var1.name)
print ("var2:",var2.name)
**************************
output():
    var1: test3/firstvar:0
	var2: test3/test4/firstvar:0
```

<font color='red'>在tf.variable_scope()中设置reuse=True可以共享变量功能</font>

**<font color='blue'>tf.reset_default_graph()</font>**

把一个图里面的变量清空

### 初始化共享变量的作用域

```python
tf.reset_default_graph()
with tf.variable_scope("test1", initializer=tf.constant_initializer(0.4) ):#(1)
    var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
    
    with tf.variable_scope("test2"):
        var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
        var3 = tf.get_variable("var3",shape=[2],initializer=tf.constant_initializer(0.3))+(2)
#########################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("var1=",var1.eval())
    print("var2=",var2.eval())
    print("var3=",var3.eval())
*******************************
out_put():
    var1= [0.4 0.4]
	var2= [0.4 0.4]
	var3= [0.3 0.3]
'''
可以在（1）和（2）两个不同位置初始化
'''
```

**<font color='green'>tf.AUTO_RESUE可以为resue自动赋值，第一次调用时resue=False,第二次调用自动变成resue=True</font>**

### 演示作用域与操作符的受限范围

```python
tf.reset_default_graph() 
with tf.variable_scope("scope1") as sp:
     var1 = tf.get_variable("v", [1])
print("sp:",sp.name)    
print("var1:",var1.name)      

with tf.variable_scope("scope2"):
    var2 = tf.get_variable("v", [1])
    
    with tf.variable_scope(sp) as sp1:#表明with ···  as sp的方式不会受外层"scope2"的影响
        var3 = tf.get_variable("v3", [1])
          
        with tf.variable_scope("") :
            var4 = tf.get_variable("v4", [1])            
print("sp1:",sp1.name)  
print("var2:",var2.name)
print("var3:",var3.name)
print("var4:",var4.name)
#################################
out_put():
    sp: scope1
	var1: scope1/v:0
	sp1: scope1
	var2: scope2/v:0
	var3: scope1/v3:0
	var4: scope1//v4:0
```

**<font color='red'>另一个限制操作符的作用域</font>**

```python
with tf.variable_scope("scope"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        x = 1.0 + v
        with tf.name_scope(""):
            y = 1.0 + v
print("v:",v.name)  
print("x.op:",x.op.name)
print("y.op:",y.op.name)
############################
out_put():
    v: scope/v:0
	x.op: scope/bar/add
	y.op: add
'''
tf.name_scope()只能限制操作符，即op，不能限制变量名，如代码中所以V不受限，X受限，另一方面可以使用空字符串将作用域返回到顶层，例如y
'''
```

## 图的基本操作

<font color='green'>tf.get_default_graph():获取当前默认图</font>

### 建立图

```python
#创建图
c = tf.constant(0.0)

g = tf.Graph()
with g.as_default():#表示用tf.Graph()函数来创建一个图
  c1 = tf.constant(0.0)
  print(c1.graph)
  print(g)
  print(c.graph)

g2 =  tf.get_default_graph()
print(g2)

tf.reset_default_graph()
g3 =  tf.get_default_graph()
print(g3)
```

```python
<tensorflow.python.framework.ops.Graph object at 0x00000299D770EFD0>
<tensorflow.python.framework.ops.Graph object at 0x00000299D770EFD0>
<tensorflow.python.framework.ops.Graph object at 0x00000299D76C15C0>
<tensorflow.python.framework.ops.Graph object at 0x00000299D76C15C0>
<tensorflow.python.framework.ops.Graph object at 0x00000299D7689D68>
```

```python
c是刚开始的默认图
g是新建的图
在g的作用域外又获取了默认图
g3是最后重设了一个图
```

### 获取张量

<font color='red'>get_tensor_by_name</font>可以获取模型里面的张量

```python
print(c1.name)
t=g.get_tensor_by_name(name=c1.name)
print(t)
############################
out_put():
    Const:0
    Tensor("Const:0", shape=(), dtype=float32)
```

### 获取节点

<font color='red'>get_operation_by_name</font>可以获取模型里面的节点

```python
# 3 获取op
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[1.0], [3.0]])

tensor1 = tf.matmul(a, b, name='exampleop')
print(tensor1.name,tensor1) 
test = g3.get_tensor_by_name("exampleop:0")
print(test)

print(tensor1.op.name)
testop = g3.get_operation_by_name("exampleop")
print(testop)

with tf.Session() as sess:
    test =  sess.run(test)
    print(test) 
    test = tf.get_default_graph().get_tensor_by_name("exampleop:0")
    print (test) 
```

```python
out_put():
    exampleop:0 Tensor("exampleop:0", shape=(1, 1), dtype=float32)
	Tensor("exampleop:0", shape=(1, 1), dtype=float32)
    exampleop
    name: "exampleop"
    op: "MatMul"
    input: "Const"
    input: "Const_1"
    attr {
      key: "T"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "transpose_a"
      value {
        b: false
      }
    }
    attr {
      key: "transpose_b"
      value {
        b: false
      }
    }

    [[7.]]
    Tensor("exampleop:0", shape=(1, 1), dtype=float32)
```

### 获取元素列表

<font color='red'>g.get_operations()</font>

