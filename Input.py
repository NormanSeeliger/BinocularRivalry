from brian2 import *
import numpy
import math

def createSpiketrains(amount,freq,duration,c,tau,csep, patching = False) :
    source_exc_1 = [1] * freq*duration + [0]*(1000-freq)*duration
    numpy.random.shuffle(source_exc_1)

    shifts1 = list()
    shifts2 = list()

    prob = math.sqrt(c)

    left_eye = list()
    for m in range(0,amount,1):
        buff_exc2 = [1]*(freq-math.trunc(prob*freq))*duration + [0]*(1000-(freq-math.trunc(prob*freq)))*duration
        numpy.random.shuffle(buff_exc2)
        for position,x in enumerate(source_exc_1):
            if prob > numpy.random.random_sample() and source_exc_1[position] == 1 and buff_exc2[position] != 1 :
                shift = math.trunc(numpy.random.exponential(tau)* numpy.random.choice([1,-1]))
                if position+shift < 0 or position+shift >= len(buff_exc2) :
                    shift = 0
                try :
                    buff_exc2[position+shift] = 1
                except IndexError :
                    print(len(buff_exc2))
                    print(shift)
                    print(position)
        left_eye.append(buff_exc2)

    right_eye = list()
    if csep > 0 :
        source_exc_2 = [1] * (freq - math.trunc(math.sqrt(csep) * freq)) * duration + [0] * (
        1000 - (freq - math.trunc(math.sqrt(csep) * freq))) * duration
        numpy.random.shuffle(source_exc_2)
        for position, x in enumerate(source_exc_1):
            if math.sqrt(csep) > numpy.random.random_sample() and source_exc_1[position] == 1 and source_exc_2[position] != 1:
                source_exc_2[position] = 1
    else :
        source_exc_2 = [1] * freq  * duration + [0] * (1000 - freq) * duration
        numpy.random.shuffle(source_exc_2)
    for m in range(0, amount, 1):
        buff_exc2 = [1] * (freq - math.trunc(prob * freq)) * duration + [0] * (
        1000 - (freq - math.trunc(prob * freq))) * duration
        numpy.random.shuffle(buff_exc2)
        for position, x in enumerate(source_exc_2):
            if prob > numpy.random.random_sample() and source_exc_2[position] == 1 and buff_exc2[position] != 1:
                shift = math.trunc(numpy.random.exponential(tau) * numpy.random.choice([1, -1]))
                if position + shift < 0 or position + shift >= len(buff_exc2):
                    shift = 0
                buff_exc2[position + shift] = 1
        right_eye.append(buff_exc2)

    # for trainleft in left_eye :
    #     print(trainleft.count(1))
    # print('Right')
    # for trainleft in right_eye :
    #     print(trainleft.count(1))
    # PATCHING
    if patching :
        print("Patching!")
        new_right_eye = list()
        for train in right_eye : # TODO : CHANGED !!!!
            tmp_list = [1] * 30 * int(duration/3) + [0] * (1000 - 10) * int(duration/3)
            numpy.random.shuffle(tmp_list)
            buffer = [0]*(len(tmp_list))
            for position, x in enumerate(tmp_list):
                if prob/1.5 > numpy.random.random_sample() and tmp_list[position] == 1 and buffer[position] != 1:
                    shift = math.trunc(numpy.random.exponential(tau) * numpy.random.choice([1, -1]))
                    if position + shift < 0 or position + shift >= len(buffer):
                        shift = 0
                    buffer[position + shift] = 1
            new_list = train[:int(duration*1000/3)] + buffer + train[int(2*duration*1000/3):]
            new_right_eye.append(new_list)
        return  [left_eye,new_right_eye]
    else :
        return [left_eye,right_eye]


def increaseContrast(amount,freq,duration,c,tau, contrast, which_eye = 0) :
    source_exc_1 = [1] * freq*duration + [0]*(1000-freq)*duration
    numpy.random.shuffle(source_exc_1)

    prob = math.sqrt(c)

    left_eye = list()
    for m in range(0,amount,1):
        buff_exc2 = [1]*(freq-math.trunc(prob*freq))*duration + [0]*(1000-(freq-math.trunc(prob*freq)))*duration
        numpy.random.shuffle(buff_exc2)
        for position,x in enumerate(source_exc_1):
            if prob > numpy.random.random_sample() and source_exc_1[position] == 1 and buff_exc2[position] != 1 :
                shift = math.trunc(numpy.random.exponential(tau)* numpy.random.choice([1,-1]))
                if position+shift < 0 or position+shift > len(buff_exc2) :
                    shift = 0
                try :
                    buff_exc2[position+shift] = 1
                except IndexError :
                    pass
        left_eye.append(buff_exc2)
    right_eye = list()
    source_exc_2 = [1] * freq  * duration + [0] * (1000 - freq) * duration
    numpy.random.shuffle(source_exc_2)
    for m in range(0, amount, 1):
        buff_exc2 = [1] * (freq - math.trunc(prob * freq)) * duration + [0] * (
        1000 - (freq - math.trunc(prob * freq))) * duration
        numpy.random.shuffle(buff_exc2)
        for position, x in enumerate(source_exc_2):
            if prob > numpy.random.random_sample() and source_exc_2[position] == 1 and buff_exc2[position] != 1:
                shift = math.trunc(numpy.random.exponential(tau) * numpy.random.choice([1, -1]))
                if position + shift < 0 or position + shift >= len(buff_exc2):
                    shift = 0
                buff_exc2[position + shift] = 1
        right_eye.append(buff_exc2)

    new_left_eye = list()
    new_right_eye = list()
    if which_eye == 1 : # left eye
        inc_period_ms = [1] * int(freq * (contrast -1) * (duration / 3)) + [0] * (1000 - int(freq * contrast) * (duration / 3))
        numpy.random.shuffle(inc_period_ms)
        print(inc_period_ms.count(1))
        print("Decreasing contrast to left eye.")
        for train in left_eye :
            inc_period = inc_period_ms[:]
            buffer = [1] * int((freq - math.trunc(prob * freq * contrast)) * int(duration/3)) + [0] * (
        1000 - int(freq - int(prob * freq * contrast)) * int(duration/3))
            for position, x in enumerate(inc_period_ms):
                if prob > numpy.random.random_sample() and inc_period_ms[position] == 1 and buffer[position] != 1:
                    shift = math.trunc(numpy.random.exponential(tau) * numpy.random.choice([1, -1]))
                    if position + shift < 0 or position + shift >= len(buffer):
                        shift = 0
                    buffer[position + shift] = 1
            new_list = train[:int(duration*1000/3)] + buffer + train[int(2*duration*1000/3):]
            new_left_eye.append(new_list)
            print(new_list.count(1))
        return [new_left_eye, right_eye]
    elif which_eye == 2 :
        print("Increasing contrast to both eyes.")
        source_exc_1 = [1] * int(freq * (contrast -1)) * int(duration/3) + [0] * (1000 - int(freq * contrast)) * int(duration/3)
        numpy.random.shuffle(source_exc_1)
        source_exc_2 = [1] * int(freq * (contrast -1)) * int(duration/3) + [0] * (1000 - int(freq * contrast)) * int(duration/3)
        numpy.random.shuffle(source_exc_2)
        for train in left_eye :
            buffer = [0] * len(source_exc_1)
            for position, x in enumerate(source_exc_1):
                if source_exc_1[position] == 1 or train[position+duration*1000/3] == 1:
                    buffer[position] = 1
            new_list = train[:int(duration*1000/3)] + buffer + train[int(2*duration*1000/3):]
            new_left_eye.append(new_list)
        for train in right_eye :
            buffer = [0] * len(source_exc_2)
            for position, x in enumerate(source_exc_2):
                if source_exc_2[position] == 1 or train[position+duration*1000/3] == 1:
                    buffer[position] = 1
            new_list = train[:int(duration*1000/3)] + buffer + train[int(2*duration*1000/3):]
            new_right_eye.append(new_list)
        return [new_left_eye, new_right_eye]

    # increase contrast to left eye