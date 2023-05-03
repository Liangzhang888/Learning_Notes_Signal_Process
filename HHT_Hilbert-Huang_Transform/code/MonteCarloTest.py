import random
import math


def montecarlo(N):
    i = 0
    count =0
    while i<=N:
        x = random.random()
        y = random.random()
        if pow(x,2) + pow(y,2) <1:
            count+=1
        i+=1
    pi = 4*count /N
    print(pi)
    print(math.pi)

if __name__ == "__main__":
    montecarlo(80000)