##flexible neuron: virtual and real layers 
Today we will talk about neurons and power series. Let's go! 

Consider the sine function. 
![dataset](https://hsto.org/webt/yc/tm/aw/yctmawjirmngqyyq7_pggbpd8g0.png)
We can expand sine in a Taylor series 
$$
	sin(x) = x-\frac{x^3}{3!}+\frac{x^5}{5!}+...+\sum_{n=1}^{\infty}\frac{(-1)^nx^{2n+1}}{(2n+1)!}
$$
Let's set n = 0
$$
	sin(x) = x
$$
![graph1](https://hsto.org/r/w1560/webt/et/tn/cn/ettncnhyazhzbxpirmthkashytq.png)
Let's set n=1
$$
	sin(x) = x-\frac{x^3}{6}
$$
![graph2](https://hsto.org/r/w1560/webt/dg/2p/mr/dg2pmrowawr8ifujv-ye6fwbtx0.png)

Let's set n = 2
$$
	sin(x) = x - \frac{x^3}{6} + \frac{x^5}{120}
$$
![graph3](https://hsto.org/r/w1560/webt/7y/xn/my/7yxnmyxwnshgznlbltvu7qtqhye.png)
Let's set n = 3
$$
	sin(x) = x - \frac{x^3}{6} + \frac{x^5}{120} - \frac{x^7}{5040}
$$
![graph4](https://hsto.org/r/w1560/webt/cw/5x/xr/cw5xxrxqcowrfh6fkxxxdkgk-ak.png)
As you can see, when we increase the order of the power series, the accuracy of the reproduced function also increases 

Let's consider a neuron with one input and one output. 

How many inputs should a neuron have with one input and one output for correct operation? 

The neuron must have two inputs. One input is x (a variable that goes to the input of the neuron) and 1(bias). 

Next question. What conversion must be done with any number to convert that number to 1? 

That's right, set any number to the zero power. 

Suppose that bias(1) is x to the zero power, we have the following expression at the output of the neuron 
$$
	y = f(w_0x^0+w_1x^1);
$$
Where f(x) is the activation function. Note that the construction 
$$
	w_0x^0+w_1x^1
$$
is very similar to the beginning of a power series. We can complicate the construction, for example,make such a thing 
$$
	y=f(w_0x^0+w_1x^1+w_2x^2);
$$
or
$$
	y=f(w_0x^0+w_1x^1+w_2x^2+w_3x^3);
$$

if we generalize this formula for one neuron with one input and one output, we get the following expression 
$$
	y=f(\sum_{n=1}^{\infty}{x^n*w_n})
$$

An approximate diagram of a neural network with two input (virtual) neurons, one output and six (real) neurons at the input of the neural network. 
![neural_net_img](https://hsto.org/r/w1560/webt/jy/1j/pi/jy1jpihbw8mgxysr1ffe4netbau.png)
x,x2,x3-is virtual inputs of the neural network ,а x^0,x^1,x^2,x2^0,x2^1,x2^2 - is real inputs(connected to weights) 
```
err2=w1*err1
err3=w2*err1
err4=w3*err1
err5=w4*err1
err6=w5*err1
err6=w6*err1
err8=err2+err3+err4
err9=err5+err6+err7
w1+=speed_edication*x^0*err2*f(x^0)*(1-f(x^0))
w2+=speed_edication*x^1*err3*f(x^0)*(f-f(x^1))
w3+=speed_edication*x^2*err2*f(x^0)*(1-f(x^0))
w4+=speed_edication*x2^0*err3*f(x2^0)*(f-f(x2^0))
w5+=speed_edication*x2^1*err2*f(x2^1)*(1-f(x2^1))
w6+=speed_edication*x2^2*err3*f(x2^2)*(f-f(x2^2))
```
where f(x) = 1/(1+exp(-x))

Let's try to summarize the obtained information. Suppose we have an input layer with n data, a hidden layer with m data and an output layer with k data. 
We create from the input layer the layer with n * power1 data. Suppose the input (virtual) layer contains the following data 
$$
INPUT=[x_1,x_2,x_3,...,x_n]
$$
Then the input (real) layer is obtained as follows 
$$
INPUT_REAL=[1,x_1,x_1^2,x_1^3,...,x_1^{power1},1,x_2,x_2^2,x_2^3,...,x_2^{power1},...,1,x_n,x_n^2,x_n^3,...,x_n^{power1}]
$$
Then comes the layer of neurons of the input layer 
$$
OUTINPUT=[neuron_1,neuron_2,...,neuron_m]
$$
Each of the neurons of the input layer receives at the input the sum of the products of the input (real)layer by the corresponding weights, and passes through the activation function.
The hidden (virtual)layer looks like: 
$$
HIDDEN=[xx_1,xx_2,xx_3,...,xx_m]
$$
It takes data from the outputs of the neurons of the input layer.
Then the hidden (real) layer is obtained as follows 
$$
HIDDENREAL=[1,xx_1,xx_1^2,xx_1^3,...,xx_1^{power2},1,xx_2,xx_2^2,xx_2^3,...,xx_2^{power2},...1,xx_m,xx_m^2,xx_m^3,...,xx_m^{power2}]
$$
This is neurons of the hidden layer 
$$
OUTHIDDEN=[neuronHidden_1,neuronHidden_2,...,neuronHidden_k]
$$
Each of the neurons of the hidden layer of the layer receives as input the sum of the products of the hidden (real) layer by the corresponding weights, and passes through the activation function.

We find the error between the actual values and the values for training the training set. 
$$
ERROR=[error_1,error_2,...,error_k]
$$

The error on the hidden (real) layer is equal: 

$$
ERRORHIDDEN = \sum_{n=1}^{k}ERROR_kW2_{j,k}
$$
where j=[0,m]

Let's ERRORHIDDEN represents an array 
$$
ERRORHIDDEN = [eh_1,eh_2,eh_3,...,eh_m]
$$

then the error on the virtual hidden layer is equal
$$
ERRORVIRTUALHIDDEN = [eh_1+eh_2+eh_{power1},eh_{power1+1}+eh_{power1+2}+...+eh_{2*power1},...,eh_{(m-1)*power1+1}+eh_{(m-1)*power1+2}+...+eh_{m*power1}]
$$
Change the weights of the input-hidden layer. 
$$
W1_{i,j}+=SPEEDEDICATION*ERRORVIRTUALHIDDEN_i*INPUTREAL_j*\frac{df(x)}{dx}_i
$$
Change the weights of the hidden-output layer. 
$$
W2_{i,j}+=SPEEDEDICATION*ERROR_i*HIDDENREAL_j*\frac{df(x)}{dx}_i
$$
where f(x) - output of neurons of the corresponding level.
SEEDEDICATION - learning rate of the neural network. 


Let's write a neural network that will classify fashion_mnist clothing sets (t-shirts, sneakers, pants, etc.). A set of 28 * 28 8-bit black-and-white picture is sent to the input of the neural network. The neural network contains a hidden layer of 300 neurons, and has 10 outputs.When training a set of clothes, each of the garments is assigned a class from 0 to 10.The neural network must learn how to correctly classify clothing items (distinguish pants from T-shirts, etc.). 

I downloaded training and test sets of clothes from github by  [refs](https://github.com/ymattu/fashion-mnist-csv)
#####I wrote a simple neural network for testing. 
```python
import numpy as np
from numpy import genfromtxt
OUT_CLASSES=10
HIDDEN_LEN=300
SPEED_EDICATION=0.0000001
EPOCHS=150000
train = genfromtxt('fashion_train.csv', delimiter=',')
train=train[1:,:]
train_x=np.array(train[:,:-1])
train_x/=255
train_y=np.array(train[:,-1])
train_y = train_y.astype(int)
out = np.zeros(OUT_CLASSES*len(train_y))
index_arr=np.reshape(train_y,(1,len(train_y)))
out = np.reshape(out,(OUT_CLASSES,len(train_y)))
np.put_along_axis(out,np.array(index_arr),1,axis=0)
train_y=out.T
test = genfromtxt('fashion_test.csv', delimiter=',')
test=test[1:,:]
test_x=np.array(test[:,:-1])
test_x/=255
test_y=np.array(test[:,-1])
test_y = test_y.astype(int)
out = np.zeros(OUT_CLASSES*len(test_y))
index_arr=np.reshape(test_y,(1,len(test_y)))
out = np.reshape(out,(OUT_CLASSES,len(test_y)))
np.put_along_axis(out,np.array(index_arr),1,axis=0)
test_y=out.T
power1=5
power2=5
weight1=2*np.random.random((power1*int(train_x.size/len(train_x)),HIDDEN_LEN))-1
weight2=2*np.random.random((HIDDEN_LEN*power2,OUT_CLASSES))-1
power1_arr = np.tile(np.arange(power1),train_x.size).reshape(train_x.size,power1)
power2_arr = np.tile(np.arange(power2),HIDDEN_LEN*len(train_x)).reshape(HIDDEN_LEN*len(train_x),power2)
power1_arr_test = np.tile(np.arange(power1),test_x.size).reshape(test_x.size,power1)
power2_arr_test = np.tile(np.arange(power2),HIDDEN_LEN*len(test_x)).reshape(HIDDEN_LEN*len(test_x),power2)
train_x2=np.reshape(train_x,(train_x.size,1))
test_x2=np.reshape(test_x,(test_x.size,1))
virtual_train_x = np.power(train_x2,power1_arr).reshape(len(train_x),int(power1*train_x.size/len(train_x)))
virtual_test_x = np.power(test_x2,power1_arr_test).reshape(len(test_x),int(power1*test_x.size/len(test_x)))
someones = np.ones(power1)
persent_train = np.ones(OUT_CLASSES).reshape(OUT_CLASSES,1)
persent_test = np.ones(OUT_CLASSES).reshape(OUT_CLASSES,1)
max_persent = 0
error_test_list=[]
error_train_list=[]
error_test_list_persent=[]
error_train_list_persent=[]
for step in range(EPOCHS):
    hidden_layer = 1/(1+np.exp(-(np.dot(virtual_train_x,weight1))))
    hidden_layer2 = np.reshape(hidden_layer,(hidden_layer.size,1))
    virtual_hidden_layer_train = np.power(hidden_layer2,power2_arr).reshape(len(hidden_layer),int(power2*hidden_layer.size/len(hidden_layer)))
    out_with_error = 1/(1+np.exp(-(np.dot(virtual_hidden_layer_train,weight2))))
    error_w2 = (train_y-out_with_error)
    weight2+=SPEED_EDICATION*virtual_hidden_layer_train.T.dot(error_w2*out_with_error*(1-out_with_error))
    error_w1 = error_w2.dot(weight2.T)
    error_w1_reshaped = np.reshape(error_w1,(int(error_w1.size/power1),power1))
    error_w1_virtual = np.reshape(np.dot(error_w1_reshaped,someones),(len(error_w1),int(error_w1.size/power1/len(error_w1))))
    weight1+=SPEED_EDICATION*virtual_train_x.T.dot(error_w1_virtual)
    hidden_layer_test = 1/(1+np.exp(-(np.dot(virtual_test_x,weight1))))
    hidden_layer2_test = np.reshape(hidden_layer_test,(hidden_layer_test.size,1))
    virtual_hidden_layer_test = np.power(hidden_layer2_test,power2_arr_test).reshape(len(hidden_layer_test),int(power2*hidden_layer_test.size/len(hidden_layer_test)))
    out_with_error_test = 1/(1+np.exp(-(np.dot(virtual_hidden_layer_test,weight2))))
    error_w2_test = (test_y-out_with_error_test)
    p_train = np.dot(np.abs(train_y-np.around(out_with_error)),persent_train)
    p_test = np.dot(np.abs(test_y-np.around(out_with_error_test)),persent_test)
    err_p_train = 100*(1-np.sum(np.logical_and(p_train,True))/len(train_y))
    err_p_test = 100*(1-np.sum(np.logical_and(p_test,True))/len(test_y))
    if max_persent<err_p_test:
	max_persent=err_p_test
    error_test_list.append(np.sum(np.square(error_w2_test))/2)
    error_train_list.append(np.sum(np.square(error_w2))/2)
    error_test_list_persent.append(err_p_test)
    error_train_list_persent.append(err_p_train)
    print("step =",step,"/",EPOCHS," error train = ",np.sum(np.square(error_w2))/2," error test = ",np.sum(np.square(error_w2_test)/2))
    print("step =",step,"/",EPOCHS," error train = ",err_p_train,"%"," error test = ",err_p_test,"%")
    print("step =",step,"/",EPOCHS," max persent test = ",max_persent,"%")
list_size = len(error_train_list)
data = error_train_list
data.extend(error_test_list)
data.extend(error_train_list_persent)
data.extend(error_test_list_persent)
data = np.reshape(data,(4,list_size))
np.savetxt('out.csv',data.T,delimiter=',',fmt='%.4f')
```
Explanations.
import the library of matrix calculations. 
```python
import numpy as np
from numpy import genfromtxt
```
define constants: the number of neural network outputs (OUT_CLASSES), the number of neurons in the hidden layer (HIDDEN_LEN), the learning rate (SPEED_EDICATION), the number of epochs (EPOCHS) 
```python
OUT_CLASSES=10
HIDDEN_LEN=300
SPEED_EDICATION=0.0000001
EPOCHS=150000
```
create a training input set (train_x) and a training output set (train_y) Similarly, create a test input set (test_x) and a test output set (test_y) 
```python
train = genfromtxt('fashion_train.csv', delimiter=',')
train=train[1:,:]
train_x=np.array(train[:,:-1])
train_x/=255
train_y=np.array(train[:,-1])
train_y = train_y.astype(int)
out = np.zeros(OUT_CLASSES*len(train_y))
index_arr=np.reshape(train_y,(1,len(train_y)))
out = np.reshape(out,(OUT_CLASSES,len(train_y)))
np.put_along_axis(out,np.array(index_arr),1,axis=0)
train_y=out.T
```

set the value of the power series 
```python
power1=5
power2=5
```
fill the weights with random values from -1 to 1 
```python
weight1=2*np.random.random((power1*int(train_x.size/len(train_x)),HIDDEN_LEN))-1
weight2=2*np.random.random((HIDDEN_LEN*power2,OUT_CLASSES))-1
```
create arrays that contain sequences from 0 to power1-1 for each element of the input array for training and test inputs, respectively (that is, for an input array of 3 elements and 2 sets and power1 = 4, this construction will create power_arr = [[0,1, 2,3,0,1,2,3,0,1,2,3], [0,1,2,3,0,1,2,3,0,1,2,3]]) 
```python
power1_arr = np.tile(np.arange(power1),train_x.size).reshape(train_x.size,power1)
power2_arr = np.tile(np.arange(power2),HIDDEN_LEN*len(train_x)).reshape(HIDDEN_LEN*len(train_x),power2)
power1_arr_test = np.tile(np.arange(power1),test_x.size).reshape(test_x.size,power1)
power2_arr_test = np.tile(np.arange(power2),HIDDEN_LEN*len(test_x)).reshape(HIDDEN_LEN*len(test_x),power2)
```

create an input layer associated with weights (virtual_train_x and virtual_test_x, respectively, for the training and test set) 
```python
train_x2=np.reshape(train_x,(train_x.size,1))
test_x2=np.reshape(test_x,(test_x.size,1))
virtual_train_x = np.power(train_x2,power1_arr).reshape(len(train_x),int(power1*train_x.size/len(train_x)))
virtual_test_x = np.power(test_x2,power1_arr_test).reshape(len(test_x),int(power1*test_x.size/len(test_x)))
```
create data in advance 
```python
someones = np.ones(power1)

persent_train = np.ones(OUT_CLASSES).reshape(OUT_CLASSES,1)
persent_test = np.ones(OUT_CLASSES).reshape(OUT_CLASSES,1)

max_persent = 0

error_test_list=[]
error_train_list=[]
error_test_list_persent=[]
error_train_list_persent=[]
```
start the learning cycle 
```python
for step in range(EPOCHS):
```
calculate the values at the output of the input layer of the neural network (which are set at the input of the hidden layer) 
```python
hidden_layer = 1/(1+np.exp(-(np.dot(virtual_train_x,weight1))))
```
create a layer from the hidden layer (hidden_layer), which is associated with the weights (virtual_hidden_layer_train) 
```python
hidden_layer2 = np.reshape(hidden_layer,(hidden_layer.size,1))
virtual_hidden_layer_train = np.power(hidden_layer2,power2_arr).reshape(len(hidden_layer),int(power2*hidden_layer.size/len(hidden_layer)))
```
calculate the output of the neurons of the hidden layer 
```python
out_with_error = 1/(1+np.exp(-(np.dot(virtual_hidden_layer_train,weight2))))
```
calculate the output error 
```python
error_w2 = (train_y-out_with_error)
```
change weights on hidden-output layers 
```python
weight2+=SPEED_EDICATION*virtual_hidden_layer_train.T.dot(error_w2*out_with_error*(1-out_with_error))
```
We find an error on the hidden real layer (error_w1), and also then we find an error on the hidden virtual layer (error_w1_virtual) 
```python
error_w1 = error_w2.dot(weight2.T)
error_w1_reshaped = np.reshape(error_w1,(int(error_w1.size/power1),power1))
error_w1_virtual = np.reshape(np.dot(error_w1_reshaped,someones),(len(error_w1),int(error_w1.size/power1/len(error_w1))))
```
change the weights of the input-hidden layer 
```python
weight1+=SPEED_EDICATION*virtual_train_x.T.dot(error_w1_virtual)
```
We calculate the error on the test set. 
```python
hidden_layer_test = 1/(1+cp.exp(-(cp.dot(virtual_test_x,weight1))))
hidden_layer2_test = cp.reshape(hidden_layer_test,(hidden_layer_test.size,1))
virtual_hidden_layer_test = cp.power(hidden_layer2_test,power2_arr_test).reshape(len(hidden_layer_test),int(power2*hidden_layer_test.size/len(hidden_layer_test)))
out_with_error_test = 1/(1+np.exp(-(np.dot(virtual_hidden_layer_test,weight2))))
error_w2_test = (test_y-out_with_error_test)
```
We get the root-mean-square error and the percentage error 
```python
p_train = cp.dot(cp.abs(train_y-cp.around(out_with_error)),persent_train)
p_test = cp.dot(cp.abs(test_y-cp.around(out_with_error_test)),persent_test)

err_p_train = 100*(1-cp.sum(np.logical_and(p_train,True))/len(train_y))
err_p_test = 100*(1-cp.sum(np.logical_and(p_test,True))/len(test_y))
```
We get the maximum error on the test set 
```python
if max_persent<err_p_test:
        max_persent=err_p_test
```
Add errors to lists and display them on the screen 
```python
error_test_list.append(cp.sum(cp.square(error_w2_test))/2)
error_train_list.append(cp.sum(cp.square(error_w2))/2)
error_test_list_persent.append(err_p_test)
error_train_list_persent.append(err_p_train)

print("step =",step,"/",EPOCHS," error train = ",np.sum(np.square(error_w2))/2," error test = ",np.sum(np.square(error_w2_test)/2))
print("step =",step,"/",EPOCHS," error train = ",err_p_train,"%"," error test = ",err_p_test,"%")
print("step =",step,"/",EPOCHS," max persent test = ",max_persent,"%")
```
After the completion of the training epochs, save the resulting lists to a file for visualization 
```python
list_size = len(error_train_list)
data = error_train_list
data.extend(error_test_list)
data.extend(error_train_list_persent)
data.extend(error_test_list_persent)
data=cp.array(data)
data = cp.ndarray.get(cp.reshape(data,(4,list_size)))


np.savetxt('out.csv',data.T,delimiter=',',fmt='%.4f')
```
Got some results. 

Used power1 = 2, power2 = 2, learning rate factor is 0.00001, number of epochs = 3000. 
![graph4](https://hsto.org/r/w1560/webt/ju/ua/xi/juuaxicrrvsqv8blkngnnvpwnko.png)
![graph5](https://hsto.org/r/w1560/webt/kh/jh/84/khjh84cfykcz_goycc52yho_1tc.png)
Got max error(percentage) of test case error = 58.6%

Used power1 = 3, power2 = 3, learning rate factor is 0.00001, number of epochs = 3000. 
![graph6](https://hsto.org/r/w1560/webt/gr/oi/ed/groied1jrjufdsr3jx68joe-wyu.png)
![graph7](https://hsto.org/r/w1560/webt/fu/cf/oz/fucfoz_jrbuyktteqcynf5ly-1g.png)

Got max error(percentage) of test case error = 61.6%

Used power1 = 4, power2 = 4, learning rate factor is 0.000001, number of epochs = 60,000. 
![graph8](https://hsto.org/r/w1560/webt/6k/h9/c1/6kh9c1rdquaz_gzhyhqg4wijgau.png)
![graph9](https://hsto.org/r/w1560/webt/fu/cf/oz/fucfoz_jrbuyktteqcynf5ly-1g.png)
Got max error(percentage) of test suite error = 62.7%

Used power1 = 5, power2 = 5, learning rate factor is 0.000001, number of epochs = 90000. 
![graph10](https://hsto.org/r/w1560/webt/l_/px/il/l_pxilkkoinxcujbs7osmtaxejc.png)
![graph11](https://hsto.org/r/w1560/webt/ci/px/_w/cipx_wiblosxgueorw2zxhza2yg.png)
Got max error(percentage) of test suite error = 62.8%
Dependence of the maximum accuracy of the test set on the power series. 
![graph12](https://hsto.org/r/w1560/webt/9l/p5/ae/9lp5ae9tibdy7n25yiopmtfxscw.png)
However, the root mean square error behaved strangely (probably due to the power series) 
![graph13](https://hsto.org/r/w1560/webt/fv/tg/1-/fvtg1-ewe4hgqejgdxgnticwnac.png)
Conclusion: The error (in percentage) of the test training set increases with increasing power series. Perhaps not quickly, by 0.1 or even less, but with an increase in the power series, the number of learning epochs and a decrease in the learning rate, it is possible to achieve an increase in the accuracy of the neural network when working with a test dataset. 

Used Books 
	1. Каниа Алексеевич Кан Нейронный сети. Эволюция
    2. Тарик Рашид.Создаем нейронную сеть.
    3. Использовалась статья Нейросеть в 11 строчек на Python 
