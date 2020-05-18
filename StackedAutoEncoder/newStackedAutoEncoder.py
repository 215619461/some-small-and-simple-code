# -*- coding: utf-8 -*-
"""
Created on Wed May 30 20:21:43 2018

@author: zy
"""

'''
去燥自编码器和栈式自编码器的综合实现
1.我们首先建立一个去噪自编码器(包含输入层四层的网络)
2.然后再对第一层的输出做一次简单的自编码压缩(包含输入层三层的网络)
3.然后再将第二层的输出做一个softmax分类
4.最后把这3个网络里的中间层拿出来，组成一个新的网络进行微调1.构建一个包含输入层的简单去噪自编码其
'''

'''
mnist Mixed National Institute of Standards and Technology database 手写数字识别
一小组图像集表示为一个四维浮点数数组, 这四个维度分别是 
[batch, height, width, channels] 通道
数量，高度，宽度，黑白为1，rgb彩色图像为3
session 会话
tensor n阶张量 = n维数组 维度a*b表示a个b维的图像 一个静态类型 rank, 和 一个 shape. n*n矩阵的阶为n阶
 [[1, 2, 3], [4, 5, 6], [7, 8, 9]] 张量为2阶，矩阵为3阶
返回的是 numpy ndarray 对象
图中的节点被称之为 op (operation 的缩写)
构建阶段和执行阶段 在构建阶段, op 的执行步骤 被描述成一个图. 在执行阶段, 使用会话执行执行图中的 op
通常在构建阶段创建一个图来表示和训练神经网络, 然后在执行阶段反复执行图中的训练 op
构建图 创建源 op (source op). 不需要任何输入,输出被传递给其它 op 做运算 配置值，配置操作
启动图 真正执行操作 创建会话session run()执行op，使用完之后释放资源或使用with
matmul 矩阵乘法 constant 设置常量 Variable 设置变量 placeholder 占位符[a,b]数量，维度
[[]] 最外面的代表这是个矩阵，里面的每个代表每行有哪些值
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os

# 读取数据
mnist = input_data.read_data_sets('../MNIST-data',one_hot=True)
'''
one_hot=True表示对label进行one-hot编码，
比如标签4可以表示为[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]。
这是神经网络输出层要求的格式。
'''

print(type(mnist))  # <class 'tensorflow.contrib.learn.python.learn.datasets.base.Datasets'>

print('Training data shape:',mnist.train.images.shape)           # Training data shape: (55000, 784) 55000行训练用数据集
print('Test data shape:',mnist.test.images.shape)                # Test data shape: (10000, 784) 10000行测试数据集
print('Validation data shape:',mnist.validation.images.shape)    # Validation data shape: (5000, 784) 5000行验证数据集
print('Training label shape:',mnist.train.labels.shape)          # Training label shape: (55000, 10) 类别

train_X = mnist.train.images    # 训练用的图片
train_Y = mnist.train.labels    # 图片对应的类别
test_X = mnist.test.images      # 测试用的图片
test_Y = mnist.test.labels


def stacked_auto_encoder():
    tf.reset_default_graph()    # 清除默认图形堆栈并重置全局默认图形
    '''
    栈式自编码器
    
    最终训练的一个网络为一个输入、一个输出和两个隐藏层
    MNIST输入(784) - > 编码层1(256)- > 编码层3(128) - > softmax分类
    
    除了输入层，每一层都用一个网络来训练，于是我们需要训练3个网络，最后再把训练好的各层组合在一起，形成第4个网络。
    '''
    
    
    '''
    网络参数定义
    '''
    n_input = 784       # 输入层节点数
    n_hidden_1 = 256    # 第一个隐藏层节点数
    n_hidden_2 = 128    # 第二个隐藏层节点数
    n_classes = 10      # 类别数
    
    learning_rate = 0.01               # 学习率
    training_epochs = 20               # 迭代轮数
    batch_size = 256                   # 小批量数量大小
    display_epoch = 10                 # 每训练display_epoch次计算average cost
    show_num = 10                      # 用于可视化
    
    savedir = "./stacked_encoder/"     #检查点文件保存路径
    savefile = 'mnist_model.cpkt'      #检查点文件名
    
    # 第一层输入，占位符 数量待定，维度为n_input
    x = tf.placeholder(dtype=tf.float32,shape=[None,n_input])
    y = tf.placeholder(dtype=tf.float32,shape=[None,n_input])
    # 每个元素被保留的概率，那么 为1就是所有元素全部保留的意思，在大量数据训练时，为了防止过拟合，添加Dropout层，设置一个0~1之间的小数
    keep_prob = tf.placeholder(dtype=tf.float32)
    
    # 第二层输入
    l2x = tf.placeholder(dtype=tf.float32,shape=[None,n_hidden_1])
    l2y = tf.placeholder(dtype=tf.float32,shape=[None,n_hidden_1])
    
    # 第三层输入
    l3x = tf.placeholder(dtype=tf.float32,shape=[None,n_hidden_2])
    l3y = tf.placeholder(dtype=tf.float32,shape=[None,n_classes])
    
    '''
    定义学习参数
    '''
    weights = {
            # 网络1 784-256-256-784
            # 产生截断正态分布随机数，取值范围为 [ mean - 2 * stddev, mean + 2 * stddev ] stddev 标准差 mean 均值，默认0
            'l1_h1':tf.Variable(tf.truncated_normal(shape=[n_input,n_hidden_1],stddev=0.1)),    # 级联使用
            'l1_h2':tf.Variable(tf.truncated_normal(shape=[n_hidden_1,n_hidden_1],stddev=0.1)),
            'l1_out':tf.Variable(tf.truncated_normal(shape=[n_hidden_1,n_input],stddev=0.1)),
            # 网络2 256-128-128-256
            'l2_h1':tf.Variable(tf.truncated_normal(shape=[n_hidden_1,n_hidden_2],stddev=0.1)), # 级联使用
            'l2_h2':tf.Variable(tf.truncated_normal(shape=[n_hidden_2,n_hidden_2],stddev=0.1)),
            'l2_out':tf.Variable(tf.truncated_normal(shape=[n_hidden_2,n_hidden_1],stddev=0.1)),
            # 网络3 128-10
            'out':tf.Variable(tf.truncated_normal(shape=[n_hidden_2,n_classes],stddev=0.1))     # 级联使用
            }
    
    biases = {
            # 网络1 784-256-256-784
            # tf.zeros([a,b]) 创建a个b维的0张量，默认为b
            'l1_b1':tf.Variable(tf.zeros(shape=[n_hidden_1])),       # 级联使用
            'l1_b2':tf.Variable(tf.zeros(shape=[n_hidden_1])),
            'l1_out':tf.Variable(tf.zeros(shape=[n_input])),
            # 网络2 256-128-128-256
            'l2_b1':tf.Variable(tf.zeros(shape=[n_hidden_2])),       # 级联使用
            'l2_b2':tf.Variable(tf.zeros(shape=[n_hidden_2])),
            'l2_out':tf.Variable(tf.zeros(shape=[n_hidden_1])),
            # 网络3 128-10
            'out':tf.Variable(tf.zeros(shape=[n_classes]))           # 级联使用
            }
    
    '''
    定义第一层网络结构  
    注意：在第一层里加入噪声，并且使用弃权层 tf.nn.dropout 784-256-256-784
    matmul 矩阵乘法 wx+b tf.nn.sigmoid 计算x的sigmoid y = 1/(1 + exp (-x))
    '''
    l1_h1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['l1_h1']),biases['l1_b1']))     
    l1_h1_dropout = tf.nn.dropout(l1_h1,keep_prob)
    l1_h2 = tf.nn.sigmoid(tf.add(tf.matmul(l1_h1_dropout,weights['l1_h2']),biases['l1_b2']))
    l1_h2_dropout = tf.nn.dropout(l1_h2,keep_prob)
    l1_reconstruction = tf.nn.sigmoid(tf.add(tf.matmul(l1_h2_dropout,weights['l1_out']),biases['l1_out']))

    # 计算代价 计算张量的各个维度上的元素的平均值
    l1_cost = tf.reduce_mean((l1_reconstruction-y)**2)

    # 定义优化器 梯度下降
    l1_optm = tf.train.AdamOptimizer(learning_rate).minimize(l1_cost)
    

    '''
    定义第二层网络结构256-128-128-256
    '''
    l2_h1 = tf.nn.sigmoid(tf.add(tf.matmul(l2x,weights['l2_h1']),biases['l2_b1']))    
    l2_h2 = tf.nn.sigmoid(tf.add(tf.matmul(l2_h1,weights['l2_h2']),biases['l2_b2']))    
    l2_reconstruction = tf.nn.sigmoid(tf.add(tf.matmul(l2_h2,weights['l2_out']),biases['l2_out']))
    
        
    # 计算代价
    l2_cost = tf.reduce_mean((l2_reconstruction-l2y)**2)

    # 定义优化器
    l2_optm = tf.train.AdamOptimizer(learning_rate).minimize(l2_cost)
    
    
    '''
    定义第三层网络结构 128-10
    '''    
    l3_logits = tf.add(tf.matmul(l3x,weights['out']),biases['out'])
    
        
    #计算代价
    l3_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l3_logits,labels=l3y))

    #定义优化器    
    l3_optm = tf.train.AdamOptimizer(learning_rate).minimize(l3_cost)
    
    
    '''
    定义级联级网络结构
    
    将前三个网络级联在一起，建立第四个网络，并定义网络结构
    '''
    # 1 联 2
    l1_l2_out = tf.nn.sigmoid(tf.add(tf.matmul(l1_h1,weights['l2_h1']),biases['l2_b1']))
    
    # 2 联 3
    logits = tf.add(tf.matmul(l1_l2_out,weights['out']),biases['out'])
    
    # 计算代价
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=l3y))

    # 定义优化器    
    optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
	# 计算大于等于改值的最小整数 ceil向上取整，floor向下取整 55000
    num_batch = int(np.ceil(mnist.train.num_examples / batch_size))
    
    #生成Saver对象，max_to_keep = 1，表名最多保存一个检查点文件，这样在迭代过程中，新生成的模型就会覆盖以前的模型。
    saver = tf.train.Saver(max_to_keep=1) 
    
    #直接载入最近保存的检查点文件
    kpt = tf.train.latest_checkpoint(savedir)
    print("kpt:",kpt)    
    
    '''
    训练 网络第一层
    '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 如果存在检查点文件 则恢复模型
        if kpt!=None:
		    # save的作用是将我们训练好的模型的参数保存下来；restore则是将训练好的参数提取出来
            saver.restore(sess, kpt) 
        print('网络第一层 开始训练')
        for epoch in range(training_epochs):
            total_cost = 0.0	# 代价或者损失loss
            for i in range(num_batch):
                batch_x,batch_y = mnist.train.next_batch(batch_size)
                # 添加噪声 每次取出来一批次的数据，将输入数据的每一个像素都加上0.3倍的高斯噪声  
                batch_x_noise = batch_x + 0.3*np.random.randn(batch_size,784)  #标准正态分布
                _,loss = sess.run([l1_optm,l1_cost],feed_dict={x:batch_x_noise,y:batch_x,keep_prob:0.5})
                # 传送数据 dropout
                total_cost += loss
                
            # 打印信息
            if epoch % display_epoch == 0:
                print('Epoch {0}/{1}  average cost {2}'.format(epoch,training_epochs,total_cost/num_batch))
                
            # 每隔1轮后保存一次检查点            
            saver.save(sess,os.path.join(savedir,savefile),global_step = epoch)
        print('训练完成')
            
        # 数据可视化                 
        test_noisy= mnist.test.images[:show_num]  + 0.3*np.random.randn(show_num,784)         
        reconstruction = sess.run(l1_reconstruction,feed_dict = {x:test_noisy,keep_prob:1.0})
        plt.figure(figsize=(1.0*show_num,1*2))	# 输出图像 指定figure的宽和高，单位为英寸
        for i in range(show_num):
            # 原始图像
            plt.subplot(3,show_num,i+1) 	# 创建单个子图  a,b,c 有a*b个图，应该展示在c位置         
            plt.imshow(np.reshape(mnist.test.images[i],(28,28)),cmap='gray')
            # a*b 数组为浮点型，值为该坐标的灰度 线性灰度色图
            plt.axis('off')		# 关闭坐标轴
            # 加入噪声后的图像
            plt.subplot(3,show_num,i+show_num*1+1)
            plt.imshow(np.reshape(test_noisy[i],(28,28)),cmap='gray')       
            plt.axis('off')
            # 去燥自编码器输出图像
            plt.subplot(3,show_num,i+show_num*2+1)
            plt.imshow(np.reshape(reconstruction[i],(28,28)),cmap='gray')       
            plt.axis('off')
        plt.show()
    
    
    '''
    训练 网络第二层
    注意：这个网络模型的输入已经不再是MNIST图片了，而是上一层网络中的一层的输出
    '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('网络第二层 开始训练')
        for epoch in range(training_epochs):
            total_cost = 0.0
            for i in range(num_batch):
                batch_x,batch_y = mnist.train.next_batch(batch_size)
                l1_out = sess.run(l1_h1,feed_dict={x:batch_x,keep_prob:1.0})
                                
                _,loss = sess.run([l2_optm,l2_cost],feed_dict={l2x:l1_out,l2y:l1_out})
                total_cost += loss
                
            #打印信息
            if epoch % display_epoch == 0:
                print('Epoch {0}/{1}  average cost {2}'.format(epoch,training_epochs,total_cost/num_batch))
                
            #每隔1轮后保存一次检查点            
            saver.save(sess,os.path.join(savedir,savefile),global_step = epoch)
        print('训练完成')
            
        #数据可视化                 
        testvec = mnist.test.images[:show_num]
        l1_out = sess.run(l1_h1,feed_dict={x:testvec,keep_prob:1.0})        
        reconstruction = sess.run(l2_reconstruction,feed_dict = {l2x:l1_out})
        plt.figure(figsize=(1.0*show_num,1*2))        
        for i in range(show_num):
            #原始图像
            plt.subplot(3,show_num,i+1)            
            plt.imshow(np.reshape(testvec[i],(28,28)),cmap='gray')   
            plt.axis('off')
            #加入噪声后的图像
            plt.subplot(3,show_num,i+show_num*1+1)
            plt.imshow(np.reshape(l1_out[i],(16,16)),cmap='gray')       
            plt.axis('off')
            #去燥自编码器输出图像
            plt.subplot(3,show_num,i+show_num*2+1)
            plt.imshow(np.reshape(reconstruction[i],(16,16)),cmap='gray')       
            plt.axis('off')
        plt.show()
    
    
    
    '''
    训练 网络第三层
    注意：同理这个网络模型的输入是要经过前面两次网络运算才可以生成
    '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('网络第三层 开始训练')
        for epoch in range(training_epochs):
            total_cost = 0.0
            for i in range(num_batch):
                batch_x,batch_y = mnist.train.next_batch(batch_size)
                l1_out = sess.run(l1_h1,feed_dict={x:batch_x,keep_prob:1.0})
                l2_out = sess.run(l2_h1,feed_dict={l2x:l1_out})
                _,loss = sess.run([l3_optm,l3_cost],feed_dict={l3x:l2_out,l3y:batch_y})
                total_cost += loss
                
            #打印信息
            if epoch % display_epoch == 0:
                print('Epoch {0}/{1}  average cost {2}'.format(epoch,training_epochs,total_cost/num_batch))
                
            #每隔1轮后保存一次检查点            
            saver.save(sess,os.path.join(savedir,savefile),global_step = epoch)
        print('训练完成')
            

        '''
        栈式自编码网络验证
        '''
        correct_prediction =tf.equal(tf.argmax(logits,1),tf.argmax(l3y,1))
        #计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))			# 把correct_prediction转换为float，计算tensor（图像）的平均值
        print('Accuracy:',accuracy.eval({x:mnist.test.images,l3y:mnist.test.labels}))	# 计算准确率 feed
    
    
    '''
    级联微调
    将网络模型联起来进行分类训练
    '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('级联微调 开始训练')
        for epoch in range(training_epochs):
            total_cost = 0.0
            for i in range(num_batch):
                batch_x,batch_y = mnist.train.next_batch(batch_size)            
                _,loss = sess.run([optm,cost],feed_dict={x:batch_x,l3y:batch_y})
                total_cost += loss
                
            #打印信息
            if epoch % display_epoch == 0:
                print('Epoch {0}/{1}  average cost {2}'.format(epoch,training_epochs,total_cost/num_batch))
            #每隔1轮后保存一次检查点            
            saver.save(sess,os.path.join(savedir,savefile),global_step = epoch)
        print('训练完成')
        print('Accuracy:',accuracy.eval({x:mnist.test.images,l3y:mnist.test.labels}))

        
if  __name__ == '__main__':
    stacked_auto_encoder()