##Гибкий нейрон: Виртуальные и Действительные слои
Сегодня мы поговорим о нейронах и степенных рядах. Поехали!

Рассмотрим функцию синуса.
![dataset](https://hsto.org/webt/yc/tm/aw/yctmawjirmngqyyq7_pggbpd8g0.png)
Разложим синус в ряд Тейлора
$$
	sin(x) = x-\frac{x^3}{3!}+\frac{x^5}{5!}+...+\sum_{n=1}^{\infty}\frac{(-1)^nx^{2n+1}}{(2n+1)!}
$$
Возьмем n = 0
$$
	sin(x) = x
$$
![graph1](https://hsto.org/r/w1560/webt/et/tn/cn/ettncnhyazhzbxpirmthkashytq.png)
возьмем n=1
$$
	sin(x) = x-\frac{x^3}{6}
$$
![graph2](https://hsto.org/r/w1560/webt/dg/2p/mr/dg2pmrowawr8ifujv-ye6fwbtx0.png)

возьмем n = 2
$$
	sin(x) = x - \frac{x^3}{6} + \frac{x^5}{120}
$$
![graph3](https://hsto.org/r/w1560/webt/7y/xn/my/7yxnmyxwnshgznlbltvu7qtqhye.png)
возьмем n = 3
$$
	sin(x) = x - \frac{x^3}{6} + \frac{x^5}{120} - \frac{x^7}{5040}
$$
![graph4](https://hsto.org/r/w1560/webt/cw/5x/xr/cw5xxrxqcowrfh6fkxxxdkgk-ak.png)
Как можно заметить,с увеличением степенного ряда также увеличивается и точность

Рассмотрим один нейрон с одним входом и одним выходом

Как вы считаете, сколько входов должно быть у нейрона с одним входом и одним выходом для корректной работы?

У этого нейрона должно быть два входа. Один вход - x (переменная, которая поступает на вход нейрона) и единичка (биас).

Следующий вопрос. Какое преобразование нужно сделать с любым числом для того,чтобы превратить это число в 1 (единичку).

Правильно, возвести любое число в нулевую степень.

Предположим,что биас (единичка) - это x в нулевой степени.Тогда мы на выходе нейрона имеем следующее выражение
$$
	y = f(w_0x^0+w_1x^1);
$$
Где f(x)-функция активации.Заметим,что конструкция
$$
	w_0x^0+w_1x^1
$$
очень похожа на начало степенного ряда.Мы можем усложнить конструкцию,например сделать такую вешь
$$
	y=f(w_0x^0+w_1x^1+w_2x^2);
$$
или
$$
	y=f(w_0x^0+w_1x^1+w_2x^2+w_3x^3);
$$

если обобщить эту формулу для одного нейрона с одним входом и одним выходом,получаем следующее выражение
$$
	y=f(\sum_{n=1}^{\infty}{x^n*w_n})
$$

Примерная схема нейросети с двумя входными (виртуальными) нейронами,одним выходным и шестью(действительными) нейронами на входе нейросети.
![neural_net_img](https://hsto.org/r/w1560/webt/jy/1j/pi/jy1jpihbw8mgxysr1ffe4netbau.png)
x,x2,x3-это виртуальные входы нейросети,а x^0,x^1,x^2,x2^0,x2^1,x2^2 - действительные (связанные с весами)
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
где f(x) = 1/(1+exp(-x))

Постараемся обобщить полученную информацию.Предположим у нас есть входной слой с n данными,скрытый слой с m данными и выходной слой с k данными.
Мы создаем от входного слоя слой с n * power1 данными.Положим,что входной(виртуальный) слой содержит следущие данные 
$$
INPUT=[x_1,x_2,x_3,...,x_n]
$$
Тогда входной(действительный) слой получается следующим образом
$$
INPUT_REAL=[1,x_1,x_1^2,x_1^3,...,x_1^{power1},1,x_2,x_2^2,x_2^3,...,x_2^{power1},...,1,x_n,x_n^2,x_n^3,...,x_n^{power1}]
$$
Потом идет слой нейронов входного слоя
$$
OUTINPUT=[neuron_1,neuron_2,...,neuron_m]
$$
Каждый из нейронов входного слоя получает на вход сумму произведений входного(действительного) слоя на соответстующие веса,и проходит через функцию активации.
Скрытый(виртуальный) слой имеет вид:
$$
HIDDEN=[xx_1,xx_2,xx_3,...,xx_m]
$$
Он принимает данные с выходов нейронов входного слоя.
Тогда скрытый(действительный) слой получается следующим образом
$$
HIDDENREAL=[1,xx_1,xx_1^2,xx_1^3,...,xx_1^{power2},1,xx_2,xx_2^2,xx_2^3,...,xx_2^{power2},...1,xx_m,xx_m^2,xx_m^3,...,xx_m^{power2}]
$$
Потом идет слой нейронов скрытого слоя слоя
$$
OUTHIDDEN=[neuronHidden_1,neuronHidden_2,...,neuronHidden_k]
$$
Каждый из нейронов скрытого слоя слоя получает на вход сумму произведений скрытого(действительного) слоя на соответстующие веса,и проходит через функцию активации.

Далее находим ошибку между действительными значениями и значениями для обучения тренировочного набора.
$$
ERROR=[error_1,error_2,...,error_k]
$$
Ошибка  на скрытом(действительном) слое образуется произведением скрытого(действительного) слоя на ошибку на выходе нейросети.В нашем случае функцией активации будет сигмоида
Ошибка на скрытом (действительном) слое равна:

$$
ERRORHIDDEN = \sum_{n=1}^{k}ERROR_kW_{j,k}
$$
где j=[0,m]

Пусть ERRORHIDDEN представляет массив
$$
ERRORHIDDEN = [eh_1,eh_2,eh_3,...,eh_m]
$$

тогда ошибка на виртуальном скрытом слое равна
$$
ERRORVIRTUALHIDDEN = [eh_1+eh_2+eh_{power1},eh_{power1+1}+eh_{power1+2}+...+eh_{2*power1},...,eh_{(m-1)*power1+1}+eh_{(m-1)*power1+2}+...+eh_{m*power1}]
$$
Изменение весов входного-скрытого слоя.
$$
W1_{i,j}+=SPEEDEDICATION*ERRORVIRTUALHIDDEN_i*INPUTREAL_j*\frac{df(x)}{dx}_i
$$
Изменение весов скрытого-выходного слоя.
$$
W2_{i,j}+=SPEEDEDICATION*ERROR_i*HIDDENREAL_j*\frac{df(x)}{dx}_i
$$
где f(x) - выход нейронов соответствующего уровня.
SEEDEDICATION - скорость обучения нейросети.


Напишем нейронную сеть,которая будет классифицировать наборы одежды fashion_mnist (футболки, кеды, штаны и т.д.). На вход нейросети подается 28*28 8-битная черно-белая картинка. Нейросеть содержит скрытый слой из 300 нейронов,и имеет 10 выходов. При обучении набора каждому из предметов одежды назначен класс от 0 до 10. Нейросеть должна научиться распознавать элементы одежды (отличать брюки от футболок и т.д.).

Я скачал тренировочный и тестовый наборы одежды с гитхаба по [ссылке](https://github.com/ymattu/fashion-mnist-csv)
#####Написал простую нейросеть для проверки.
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
Пояснения.
импортируем библиотеку матричных вычислений.
```
import numpy as np
from numpy import genfromtxt
```
определяем константы.Это число выходов нейросети(OUT_CLASSES),количество нейронов в скрытом слою (HIDDEN_LEN),скорость обучения(SPEED_EDICATION),количество эпох(EPOCHS)
```
OUT_CLASSES=10
HIDDEN_LEN=300
SPEED_EDICATION=0.0000001
EPOCHS=150000
```
создаем тренировочный набор входных данных(train_x) и тренировочный набор выходных данных(train_y).Аналогичным образом создаем тестовый набор входных данных(test_x) и тестовый набор выходных данных(test_y)
```
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

устанавливаем значение степенных рядов
```
power1=5
power2=5
```
заполняем значения весов случайными значениями от -1 до 1
```
weight1=2*np.random.random((power1*int(train_x.size/len(train_x)),HIDDEN_LEN))-1
weight2=2*np.random.random((HIDDEN_LEN*power2,OUT_CLASSES))-1
```
создаем массивы,которые содержат последовательности от 0 до power1-1 для каждого элемента входного массива для тренировочного и тестового входов соответственно(то есть для входного массива из 3 элементов и 2 наборов и power1=4 эта конструкция создаст power_arr=[[0,1,2,3,0,1,2,3,0,1,2,3],[0,1,2,3,0,1,2,3,0,1,2,3]])
```
power1_arr = np.tile(np.arange(power1),train_x.size).reshape(train_x.size,power1)
power2_arr = np.tile(np.arange(power2),HIDDEN_LEN*len(train_x)).reshape(HIDDEN_LEN*len(train_x),power2)
power1_arr_test = np.tile(np.arange(power1),test_x.size).reshape(test_x.size,power1)
power2_arr_test = np.tile(np.arange(power2),HIDDEN_LEN*len(test_x)).reshape(HIDDEN_LEN*len(test_x),power2)
```

создаем связанный с весами слой(virtual_train_x и virtual_test_x соответственно для тренировочного и тестового набора)
```
train_x2=np.reshape(train_x,(train_x.size,1))
test_x2=np.reshape(test_x,(test_x.size,1))
virtual_train_x = np.power(train_x2,power1_arr).reshape(len(train_x),int(power1*train_x.size/len(train_x)))
virtual_test_x = np.power(test_x2,power1_arr_test).reshape(len(test_x),int(power1*test_x.size/len(test_x)))
```
предварительно создаем данные
```
someones = np.ones(power1)

persent_train = np.ones(OUT_CLASSES).reshape(OUT_CLASSES,1)
persent_test = np.ones(OUT_CLASSES).reshape(OUT_CLASSES,1)

max_persent = 0

error_test_list=[]
error_train_list=[]
error_test_list_persent=[]
error_train_list_persent=[]
```
начинаем эпохи обучения
```
for step in range(EPOCHS):
```
вычисляем значения на выходе входного слоя нейросети(которые поступают на вход скрытого слоя)
```
hidden_layer = 1/(1+np.exp(-(np.dot(virtual_train_x,weight1))))
```
создаем из скрытого слоя(hidden_layer) слой,который связан с весами(virtual_hidden_layer_train)
```
hidden_layer2 = np.reshape(hidden_layer,(hidden_layer.size,1))
virtual_hidden_layer_train = np.power(hidden_layer2,power2_arr).reshape(len(hidden_layer),int(power2*hidden_layer.size/len(hidden_layer)))
```
вычисляем выход нейронов скрытого слоя
```
out_with_error = 1/(1+np.exp(-(np.dot(virtual_hidden_layer_train,weight2))))
```
вычисляем ошибку на выходе
```
error_w2 = (train_y-out_with_error)
```
корректируем весы на выходных - скрытых слоях
```
weight2+=SPEED_EDICATION*virtual_hidden_layer_train.T.dot(error_w2*out_with_error*(1-out_with_error))
```
Находим ошибку на скрытом действительном слое(error_w1),а также потом находим ошибку на скрытом виртуальном слое(error_w1_virtual)
```
error_w1 = error_w2.dot(weight2.T)
error_w1_reshaped = np.reshape(error_w1,(int(error_w1.size/power1),power1))
error_w1_virtual = np.reshape(np.dot(error_w1_reshaped,someones),(len(error_w1),int(error_w1.size/power1/len(error_w1))))
```
корректируем веса входного - скрытого слоя
```
weight1+=SPEED_EDICATION*virtual_train_x.T.dot(error_w1_virtual)
```
Вычисляем ошибку на тестовом наборе.
```
hidden_layer_test = 1/(1+cp.exp(-(cp.dot(virtual_test_x,weight1))))
hidden_layer2_test = cp.reshape(hidden_layer_test,(hidden_layer_test.size,1))
virtual_hidden_layer_test = cp.power(hidden_layer2_test,power2_arr_test).reshape(len(hidden_layer_test),int(power2*hidden_layer_test.size/len(hidden_layer_test)))
out_with_error_test = 1/(1+np.exp(-(np.dot(virtual_hidden_layer_test,weight2))))
error_w2_test = (test_y-out_with_error_test)
```
Находим среднеквадратическую погрешность и ошибку в процентах
```
p_train = cp.dot(cp.abs(train_y-cp.around(out_with_error)),persent_train)
p_test = cp.dot(cp.abs(test_y-cp.around(out_with_error_test)),persent_test)

err_p_train = 100*(1-cp.sum(np.logical_and(p_train,True))/len(train_y))
err_p_test = 100*(1-cp.sum(np.logical_and(p_test,True))/len(test_y))
```
Находим максимальную ошибку на тестовом наборе
```
if max_persent<err_p_test:
        max_persent=err_p_test
```
Добавляем ошибки в списки и выводим их на экран
```
error_test_list.append(cp.sum(cp.square(error_w2_test))/2)
error_train_list.append(cp.sum(cp.square(error_w2))/2)
error_test_list_persent.append(err_p_test)
error_train_list_persent.append(err_p_train)

print("step =",step,"/",EPOCHS," error train = ",np.sum(np.square(error_w2))/2," error test = ",np.sum(np.square(error_w2_test)/2))
print("step =",step,"/",EPOCHS," error train = ",err_p_train,"%"," error test = ",err_p_test,"%")
print("step =",step,"/",EPOCHS," max persent test = ",max_persent,"%")
```
После завершения эпох обучения сохраняем полученные списки в файл для визуализации
```
list_size = len(error_train_list)
data = error_train_list
data.extend(error_test_list)
data.extend(error_train_list_persent)
data.extend(error_test_list_persent)
data=cp.array(data)
data = cp.ndarray.get(cp.reshape(data,(4,list_size)))


np.savetxt('out.csv',data.T,delimiter=',',fmt='%.4f')
```
Получил некоторые результаты.

Использовал power1=2,power2=2,коэффициент скорости обучения равен 0.00001,количество эпох=3000.
![graph4](https://hsto.org/r/w1560/webt/ju/ua/xi/juuaxicrrvsqv8blkngnnvpwnko.png)
![graph5](https://hsto.org/r/w1560/webt/kh/jh/84/khjh84cfykcz_goycc52yho_1tc.png)
Получил максимальную ошибку(в процентах) тестового набора error=58.6%

Использовал power1=3,power2=3,коэффициент скорости обучения равен 0.00001,количество эпох=3000.
![graph6](https://hsto.org/r/w1560/webt/gr/oi/ed/groied1jrjufdsr3jx68joe-wyu.png)
![graph7](https://hsto.org/r/w1560/webt/fu/cf/oz/fucfoz_jrbuyktteqcynf5ly-1g.png)

Получил максимальную ошибку(в процентах) тестового набора error=61,6%

Использовал power1=4,power2=4,коэффициент скорости обучения равен 0.000001,количество эпох=60000.
![graph8](https://hsto.org/r/w1560/webt/6k/h9/c1/6kh9c1rdquaz_gzhyhqg4wijgau.png)
![graph9](https://hsto.org/r/w1560/webt/fu/cf/oz/fucfoz_jrbuyktteqcynf5ly-1g.png)
Получил максимальную ошибку(в процентах) тестового набора error=62.7%

Использовал power1=5,power2=5,коэффициент скорости обучения равен 0.000001,количество эпох=90000.
![graph10](https://hsto.org/r/w1560/webt/l_/px/il/l_pxilkkoinxcujbs7osmtaxejc.png)
![graph11](https://hsto.org/r/w1560/webt/ci/px/_w/cipx_wiblosxgueorw2zxhza2yg.png)
Получил максимальную ошибку (в процентах) тестового набора error=62.8%
Зависимость максиальной точности тестового набора от степенного ряда.
![graph12](https://hsto.org/r/w1560/webt/9l/p5/ae/9lp5ae9tibdy7n25yiopmtfxscw.png)
Однако среднеквадратическая ошибка вела себя странно (наверно из-за степенного ряда)
![graph13](https://hsto.org/r/w1560/webt/fv/tg/1-/fvtg1-ewe4hgqejgdxgnticwnac.png)
Вывод: Ошибка (в процентах) тестового обучающего набора при увеличении степенного ряда растет. Возможно, не быстро, на 0.1 или даже меньше, но с увеличением степенного ряда, количества эпох обучения и уменьшением скорости обучения можно добиться увеличения точности нейросети при работе с тестовым набором данных.

Используемая литература
	1. Каниа Алексеевич Кан Нейронный сети. Эволюция
    2. Тарик Рашид.Создаем нейронную сеть.
    3. Использовалась статья Нейросеть в 11 строчек на Python 
