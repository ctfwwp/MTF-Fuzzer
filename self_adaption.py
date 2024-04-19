import math
def adaption_factor(direction,x):
    #init_value  初始值，0<init_value<1
    #direction   增加或者减小的方向  1 or -1
    #number 记录某种类型测试用例的个数
    k = 0.3
    num = 16000

    if direction == 1:   # n = 60
        init_value = k * math.exp(-((2*x-2)/(num-2)))
        # init_value = init_value * (0.5 ** (number / 400))
        #init_value = 0.2
    elif direction == 0:  # n = 30
        init_value = k * math.exp(-((2*x-2)/(num-2)))
        # init_value = init_value * (0.5 ** (number / 400))
       # init_value = 0.2
    elif direction == -1:   # n = 35
        init_value = k * math.exp(-((2*x-2)/(num-2)))
        # init_value = init_value * (0.5 ** (number / 200))
        #init_value = 0.2
    return init_value



