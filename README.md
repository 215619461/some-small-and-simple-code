# some-small-and-simple-code
这是一些简单但是有一点意思的代码  
  
  
StackedAutoEncoder是栈式自编码实现识别mnist数据集  
  
MNISTWithTf.js 是用tensorflow.js实现在线训练识别mnist数据集并保存模型，加载已训练的模型，利用训练好的模型进行在线手写数字识别三个部分。其中训练模型采取了两层cnn结构，都由0.25的dropout，第一层还加入了高斯噪声。训练之后还可以加载训练好的模型继续训练。我自己训练了30次之后准确率为99.39%
