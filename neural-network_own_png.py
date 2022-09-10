import glob

import imageio
import matplotlib
import matplotlib.pyplot
import numpy
import torch
import scipy.special
# 神经网络类定义
def relu(a):
    return (numpy.maximum(0,a))
class neuralNetwork:

    # 初始化神经网络
    def __init__(self,inputondes,hiddennodes,outputnodes,learningrate):

        # 设置输入，隐藏，输出神经元个数
        self.inodes = inputondes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        #学习率
        self.lr = learningrate

        #权值矩阵wih和who
        #w11 w21
        #w12 w22 etc
        ##这里以隐藏节点为行，输入节点为列，可避免下一步计算乘法时的转置操作
        self.wih =torch.randn((self.hnodes,self.inodes))
        self.who =torch.randn((self.onodes,self.hnodes))

        # 激活函数设置为sigmoid函数
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # 训练神经网络
    def train(self, inputs_list, targets_list):
        # 将输入列表转化为矩阵形式并转置
        inputs = numpy.array(inputs_list, ndmin = 2).T
        targets = numpy.array(targets_list, ndmin =2).T

        ##神经网络计算过程
        #隐藏层计算
        hidden_inputs =numpy.dot(self.wih, inputs)
        #对隐藏层应用激活函数
        hidden_outputs = self.activation_function(hidden_inputs)
        #输出层计算
        final_inputs = numpy.dot(self.who,hidden_outputs)
        #最终输出

        final_outputs =self.activation_function(final_inputs)

        #误差=target-final-outputs
        output_errors =targets-final_outputs

        #隐藏层误差是 output_errors，按权重拆分误差，在隐藏节点处重新组合
        hidden_errors =numpy.dot(self.who.T, output_errors)


        #更新隐藏层和输出层的权值
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs))


        pass
    # 测试神经网络
    def query(self,input_list):
        # 将输入列表转化为二维数组
        inputs = numpy.array(input_list,ndmin = 2).T

        #隐藏层计算
        hidden_inputs = numpy.dot(self.wih, inputs)

        #隐藏层计算结果通过激活函数进行处理
        hidden_outputs = self.activation_function(hidden_inputs)

        #输出层计算
        final_inputs =numpy.dot(self.who, hidden_outputs)
        #计算最终输出
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

# 设置各层神经元个数
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
# 学习率
learning_rate = 0.1
#创建神经网络实例
n =neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
#读取训练集并对训练数据进行处理
training_data_file=open("mnist_train.csv",'r')
training_data_list=training_data_file.readlines()
training_data_file.close()

#训练神经网络
epochs = 7  #训练次数
for e in range(epochs):
    for record in training_data_list:
        all_value = record.split(',')
        inputs = (numpy.asfarray(all_value[1:])/255.0 *0.99)+0.01
        targets = numpy.zeros(output_nodes) +0.01
        targets[int(all_value[0])] =0.99
        n.train(inputs,targets)

#读取测试集
test_data_file =open("mnist_test.csv",'r')
test_data_list =test_data_file.readlines()
test_data_file.close()
#测试神经网络
scorecard =[]  #记分列表

for record in test_data_list:
    #测试集数据处理
    all_value=record.split(',')
    correct_label =int(all_value[0])
    inputs =(numpy.asfarray(all_value[1:])/255.0 *0.99)+0.01
    #测试
    outputs=n.query(inputs)
    #取测试结果数组中最大值
    label =numpy.argmax(outputs)
    #检查预测值与真实值是否相等并计分
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

#计算神经网络预测表现：正确率
scorecard_array = numpy.asfarray(scorecard)
accuracy_rate =scorecard_array.sum()/scorecard_array.size
print(accuracy_rate)

# 自己的手写数字
our_own_dataset = []
# 加载图片
for image_file_name in glob.glob('own_png/?.png'):
    # 使用文件名设置正确的标签
    label = int(image_file_name[-5:-4])
    print(label)

    # 将 png文件中的图像数据加载到数组中
    # print("loading ... ", image_file_name)
    img_array = imageio.imread(image_file_name, as_gray=True)

    # 从 28x28 重塑为 784 个值的列表(这样做的原因是，常规而言，0指的是黑色，255指的是白色，但是，MNIST数据集使用相反的方式表示，
    # 因此不得不将值逆转过来以匹配MNIST数据。)
    img_data = 255.0 - img_array.reshape(784)

    # 处理数据
    img_data = (img_data / 255.0 * 0.99) + 0.01
    # print(numpy.min(img_data))
    # print(numpy.max(img_data))

    # 将标签和图像数据附加到测试数据集
    record = numpy.append(label, img_data)
    our_own_dataset.append(record)

    pass

# 用自己的图片测试神经网络


score = []
for item in range(7):
    #显示
    matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()
    # 正确答案是第一个值
    correct_label = our_own_dataset[item][0]
    # 输入的是剩余值
    inputs = our_own_dataset[item][1:]

    # 测试神经网络
    outputs = n.query(inputs)
    print (outputs)

    # 最大值的索引就是标签
    label = numpy.argmax(outputs)
    print("network says ", label)
    # 是否正确
    if (label == correct_label):
        print ("yes!")
        score.append(1)
    else:
        print ("no!")
        score.append(0)
        pass
score = numpy.asfarray(score)
A_rate =score.sum()/score.size
print("正确率为：",A_rate)
