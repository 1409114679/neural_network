import numpy
import scipy.special
import matplotlib.pyplot
import imageio
import glob
import torch
# 神经网络类定义
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
        self.wih =numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes, self.inodes))
        self.who =numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.onodes,self.hnodes))


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
epochs = 6  #训练次数
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





