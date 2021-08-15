import numpy as np
from numpy import genfromtxt
import cupy as cp
import mpmath as mp
OUT_CLASSES=10
HIDDEN_LEN=300
SPEED_EDICATION=0.000001
EPOCHS=3000
power1=3
power2=3


train = genfromtxt('fashion_train.csv', delimiter=',')
train=train[1:,:]
train_x=np.array(train[:,:-1])
train_x/=255
train_y=np.array(train[:,-1])


#a = np.array([[0,0,0,0],
#                [0,0,0,0],
#                [0,0,0,0]])
#np.put_along_axis(a,np.array([[0,2,1,0]]),1,axis = 0)
#print(a)

train_y = train_y.astype(int)
out = np.zeros(OUT_CLASSES*len(train_y))
index_arr=np.reshape(train_y,(1,len(train_y)))


out = np.reshape(out,(OUT_CLASSES,len(train_y)))
np.put_along_axis(out,np.array(index_arr),1,axis=0)
train_y=cp.array(out.T)

test = genfromtxt('fashion_test.csv', delimiter=',')
test=test[1:,:]
test_x=np.array(test[:,:-1])
test_x/=255
test_y=np.array(test[:,-1])

test_y = np.array(test_y.astype(int))
out = np.zeros(OUT_CLASSES*len(test_y))
index_arr=np.reshape(test_y,(1,len(test_y)))


out = np.reshape(out,(OUT_CLASSES,len(test_y)))
np.put_along_axis(out,np.array(index_arr),1,axis=0)
test_y=cp.array(out.T)


weight1=cp.array(2*np.random.random((power1*int(train_x.size/len(train_x)),HIDDEN_LEN))-1)
weight2=cp.array(2*np.random.random((HIDDEN_LEN*power2,OUT_CLASSES))-1)

power1_arr = np.tile(np.arange(power1),train_x.size).reshape(train_x.size,power1)
power2_arr = np.tile(np.arange(power2),HIDDEN_LEN*len(train_x)).reshape(HIDDEN_LEN*len(train_x),power2)

power1_arr_test = np.tile(np.arange(power1),test_x.size).reshape(test_x.size,power1)
power2_arr_test = cp.array(np.tile(np.arange(power2),HIDDEN_LEN*len(test_x)).reshape(HIDDEN_LEN*len(test_x),power2))

train_x2=np.reshape(train_x,(train_x.size,1))
test_x2=np.reshape(test_x,(test_x.size,1))

virtual_train_x = cp.array(np.power(train_x2,power1_arr).reshape(len(train_x),int(power1*train_x.size/len(train_x))))

virtual_test_x = cp.array(np.power(test_x2,power1_arr_test).reshape(len(test_x),int(power1*test_x.size/len(test_x))))
#print(len(virtual_train_x),virtual_train_x.size/len(virtual_train_x))

someones = cp.array(np.ones(power1))

persent_train = cp.array(np.ones(OUT_CLASSES).reshape(OUT_CLASSES,1))
persent_test = cp.array(np.ones(OUT_CLASSES).reshape(OUT_CLASSES,1))

max_persent = 0

error_test_list=[]
error_train_list=[]
error_test_list_persent=[]
error_train_list_persent=[]


for step in range(EPOCHS):

    hidden_layer = 1/(1+cp.exp(-(cp.dot(virtual_train_x,weight1))))
    hidden_layer2 = cp.reshape(hidden_layer,(hidden_layer.size,1))
    virtual_hidden_layer_train = cp.power(hidden_layer2,cp.array(power2_arr)).reshape(len(hidden_layer),int(power2*hidden_layer.size/len(hidden_layer)))
    out_with_error = 1/(1+cp.exp(-(cp.dot(virtual_hidden_layer_train,cp.array(weight2)))))
    error_w2 = (cp.array(train_y)-out_with_error)
    weight2+=SPEED_EDICATION*virtual_hidden_layer_train.T.dot(error_w2*out_with_error*(1-out_with_error))
    error_w1 = error_w2.dot(weight2.T)

    error_w1_reshaped = cp.reshape(error_w1,(int(error_w1.size/power1),power1))
    error_w1_virtual = cp.reshape(cp.dot(error_w1_reshaped,someones),(len(error_w1),int(error_w1.size/power1/len(error_w1))))

    #print(len(error_w1_virtual),error_w1_virtual.size/len(error_w1_virtual))

    weight1+=SPEED_EDICATION*virtual_train_x.T.dot(error_w1_virtual)




    hidden_layer_test = 1/(1+cp.exp(-(cp.dot(virtual_test_x,weight1))))
    hidden_layer2_test = cp.reshape(hidden_layer_test,(hidden_layer_test.size,1))
    virtual_hidden_layer_test = cp.power(hidden_layer2_test,power2_arr_test).reshape(len(hidden_layer_test),int(power2*hidden_layer_test.size/len(hidden_layer_test)))
    out_with_error_test = 1/(1+cp.exp(-(np.dot(virtual_hidden_layer_test,weight2))))
    error_w2_test = (test_y-out_with_error_test)

    p_train = cp.dot(cp.abs(train_y-cp.around(out_with_error)),persent_train)
    p_test = cp.dot(cp.abs(test_y-cp.around(out_with_error_test)),persent_test)

    err_p_train = 100*(1-cp.sum(np.logical_and(p_train,True))/len(train_y))
    err_p_test = 100*(1-cp.sum(np.logical_and(p_test,True))/len(test_y))

    if max_persent<err_p_test:
        max_persent=err_p_test

    
    error_test_list.append(cp.sum(cp.square(error_w2_test))/2)
    error_train_list.append(cp.sum(cp.square(error_w2))/2)
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
data=cp.array(data)
data = cp.ndarray.get(cp.reshape(data,(4,list_size)))
#data = data.get()
#data = np.reshape(data,(4,list_size))

np.savetxt('out.csv',data.T,delimiter=',',fmt='%.4f')


#pow1=3
#a = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
#b = np.ones(pow1)

#c = np.reshape(a,(int(a.size/pow1),pow1))
#print(c)
#print(np.reshape(np.dot(c,b),(len(a),int(a.size/pow1/len(a)))))




#end for
#print(len(np.dot(virtual_train_x,weight1)),np.dot(virtual_train_x,weight1).size/len(np.dot(virtual_train_x,weight1)))

#a2 = np.array([[1,2,3,4],[5,6,7,8]])
#print(len(a2))
#pow1 = 5
#print(np.tile(np.arange(pow1),a2.size).reshape(a2.size,pow1))
#pow1_arr = np.tile(np.arange(pow1),a2.size).reshape(a2.size,pow1)
#a22 = np.reshape(a2,(a2.size,1))
#print(np.power(a22,pow1_arr))
#print(np.power(a22,pow1_arr).reshape(len(a2),int(pow1*a2.size/len(a2))))


#print(a22)
#power = np.array([[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2]])
#print(np.arange(pow1))
#print(np.power(a22,power))
#print(a2.size)
#b2 = np.array([[0,1,2],[0,1,2],[0,1,2],[0,1,2]])
#aa2=np.ones((4,1))
#E2=np.eye(4)
#print(np.dot(a2,E2))
#aaa2 = np.dot(np.dot(a2,E2),aa2)
#print(aaa2)
#print(np.power(aaa2,b2))

