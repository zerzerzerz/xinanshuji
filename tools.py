import random 
from collections import Counter
import time
import math
from functools import wraps
import json
# =====================================================================================
# 注释掉两条分割线之间的东西来使用


# import numpy as np 
# import torch
# from torch import nn
# import pandas as pd
# import click
# # import time
# # from functools import wraps


# # 当我对func函数装饰的时候，自然需要传入func作为实参
# # @add_time 等价于 add_time(func),因此我有一个新的函数g接住add_time的返回值，这里假设返回的函数是wrap_func
# # 当我执行g的时候，实际上执行的是wrap_func，而执行wrap_func实际上主体在执行func，因此wrap_func的参数需要传func的参数
# def add_time(func):
#     # 这个wraps是固定写法，为了函数文档和函数名字的不变
#     @wraps(func)
#     def wrap_func(*args,**kwds):
#         s = time.time()
#         # 这是我的函数的主体部分，我希望在前后加上一些东西
#         ans = func(*args,**kwds)
#         e = time.time()
#         print(f'Time for {func.__name__} is {e - s}s')
#         return ans
#     return wrap_func



# class BatchNorm(nn.Module):
#     '''
#     input X is (B,C,H,W)
#     '''
#     def __init__(self):
#         super().__init__()

#     def forward(self,X):
#         return (X - X.mean(dim=(0,2,3),keepdim=True)) / X.std(dim=(0,2,3),keepdim=True)

# class LayerNorm(nn.Module):
#     '''
#     input X is (B,C,H,W)
#     '''
#     def __init__(self):
#         super().__init__()

#     def forward(self,X):
#         return (X - X.mean(dim=(1,2,3),keepdim=True)) / X.std(dim=(1,2,3),keepdim=True)

# class InstanceNorm(nn.Module):
#     '''
#     input X is (B,C,H,W)
#     '''
#     def __init__(self):
#         super().__init__()

#     def forward(self,X):
#         return (X - X.mean(dim=(2,3),keepdim=True)) / X.std(dim=(2,3),keepdim=True)


# def shape_conv(h,h_k,stride,padding):
#     return int((h - h_k + stride + 2 * padding) / stride)

# def shape_trans_conv(h,h_k,stride,padding,out_padding=0):
#     """
#         H_out = (H_in - 1)*stride - 2*padding + kernel_size + out_padding
#         If we choose stride = 2,kernel_size = 2 and both padding are 0
#         Then the H and W are doubled
#     """
#     return int(
#         (h - 1) * stride - 2 * padding + h_k + out_padding
#     )


# def draw_table(m,mode):
#     """
#     mode = 'add' or 'mul'
#     """

#     sheet = np.zeros((m,m))
#     if mode == 'add':
#         for i in range(m):
#             for j in range(m):
#                 sheet[i,j] = (i + j) % m
#     else:
#         for i in range(m):
#             for j in range(m):
#                 sheet[i,j] = (i * j) % m

#     ans = pd.DataFrame(sheet)
#     ans.to_excel(f'm={m}_mode={mode}.xlsx')

# class PixelNorm(nn.Module):
#     def __init__(self):
#         super(PixelNorm,self).__init__()
    
#     def forward(self,X):
#         tmp = X * X
#         return X / torch.sqrt(tmp.sum(dim=1,keepdim=True))

# class Maxout(nn.Module):
#     def __init__(self,num_in,num_out,pieces):
#         super(Maxout,self).__init__()
#         self.W = nn.Parameter(torch.randn(num_in,num_out,pieces))
#         self.b = nn.Parameter(torch.randn(num_out,pieces))
#     def forward(self,X):
#         return torch.from_numpy(np.max(np.tensordot(X.detach().numpy(),self.W.detach().numpy(),axes=1) + self.b.detach().numpy(),axis=2))




# =====================================================================================



def add_time(func):
    # 这个wraps是固定写法，为了函数文档和函数名字的不变
    @wraps(func)
    def wrap_func(*args,**kwds):
        s = time.time()
        # 这是我的函数的主体部分，我希望在前后加上一些东西
        ans = func(*args,**kwds)
        e = time.time()
        print(f'Time for {func.__name__} is {e - s}s')
        return ans
    return wrap_func


def __lcm(a:int,b:int) -> int:
    return a*b//gcd(a,b)

def lcm(*a):
    '''lcm(3,4,5,6,7)'''
    assert len(a) > 1,f'The number of input of function lcm must be bigger than 1'
    ans = __lcm(a[0],a[1])
    for item in a[2:]:
        ans = __lcm(ans,item)
    return ans
    pass



def __gcd(a,b):
    a = abs(a)
    b = abs(b)
    if a == 0:
        return b
    if b == 0:
        return a

    while a!=0 and b!=0:
        a,b = b,a%b

    if b == 0:
        return a
    if a == 0:
        return b
    


def gcd(*a):
    '''
    求多个整数的最大公因数
    gcd(12,32,45,64)
    '''
    # for item in a:
    #     if item == 1:
    #         return 1
    # if len(a) == 2:
    #     return __gcd(a[0],a[1])
    # else:
    #     d = __gcd(a[0],a[1])
    #     for i in range(2,len(a)):
    #         d = __gcd(d,a[i])
    #         if d == 1:
    #             return d
    #     return d
    l = len(a)
    assert l>1,f'The length of list should be bigger than 1'
    gcd = 1
    for i in range(l-1):
        x = abs(a[i])
        y = abs(a[i+1])
        while x!=0 and y!=0:
            x,y = y,x%y
        if x == 0:
            gcd = y
        else:
            gcd = x
        if gcd == 1:
            return gcd
    return gcd


def is_prime (n):
    n = int(n)
    if n == 1:
        return False

    if n > 1e6:
        return is_large_prime(int(n))
    upper = int(n ** 0.5)
    upper += 1
    for i in range(2,upper):
        if n % i == 0 :
            return False
    return True


def get_prime (N):
    '''获取前小于等于N的所有素数'''
    a = [True] * (N + 1)
    indices = list(range(2,N + 1))
    for index in indices:
        if not a[index]:
            pass 
        else :
            i = 2
            while index * i <= N :
                a[i * index] = False 
                i += 1
    ans = list(filter(lambda x:a[x],indices))
    return ans


def bezout(a,b):
    '''计算贝祖等式'''
    s2 = 0
    s1 = 1
    t2 = 1
    t1 = 0
    q = int(a / b)
    r2 = a % b
    r1 = b

    while r2 != 0:
        s2,s1 = -q * s2 + s1 , s2
        t2,t1 = -q * t2 + t1 , t2
        q = int(r1 / r2)
        r2,r1 = -q * r2 + r1 , r2
    return (s2,t2)




# @add_time
def get_inverse(a,m):
    '''求解a模m的逆元'''
    tmp = bezout(a,m)[0]
    while tmp <= 0:
        tmp += m
    return tmp


def china_res(b:list,m:list):
    '''
    the first element of returned tuple is the ANS\n
    the second element of returned tuple is the product of M_i\n
    '''
    M = 1
    for item in m:
        M *= item
    
    ans = 0
    
    for i in range(len(m)):
        m_i = m[i]
        M_i = int(M / m_i)
        M_i_inverse = get_inverse(M_i,m_i)
        ans += int(b[i] * M_i * M_i_inverse)
    
    # ans %= M
    ans = ans % M
    # print(ans)
    # print(M)
    return (ans,M)

def ten2two(n:int,total_bits:int=8):
    '''n should be positive'''
    sign = (n >= 0)
    n = abs(n)
    tmps = []
    while n != 0:
        tmps.append(int(n&1))
        n >>= 1
    tmps.append(0)
    if not sign:
        for i in range(len(tmps)):
            tmps[i] = 1 if tmps[i] == 0 else 0
        tmps[0] += 1
        for i in range(len(tmps)):
            if tmps[i] == 2:
                tmps[i] = 0
                if i < len(tmps) - 1:
                    tmps[i + 1] += 1
                else:
                    tmps.append(1)
    tmps = tmps[::-1]
    tmps = ''.join(str(item) for item in tmps)
    tmps = tmps[0] + tmps[0]*(total_bits - len(tmps)) + tmps[1:] if total_bits >= len(tmps) else tmps[len(tmps) - total_bits:]
    return tmps


def euler_function(n:int):
    '''求n的欧拉函数phi(n)'''
    N = n
    primes = get_prime(n)
    p_set = []
    for p in primes:
        while n % p == 0:
            n /= p
            p_set.append(p)
    p_set = set(p_set)
    ans = N
    for p in p_set:
        ans *= (1 - 1/p)
    return int(ans)


def solve_foce(a:int,b:int,m:int):
    '''
    solve_first_order_congruence_equation
    return value is a tuple
    the first element of tuple is constant
    the second element of tuple is the coefficient of t
    '''
    gcd_a_m = gcd(a,m)
    a1 = get_inverse(int(a/gcd_a_m), int(m/gcd_a_m))
    a2 = b / gcd_a_m * a1
    a2 = int(a2)
    return (a2,int(m/gcd_a_m))



# @add_time
def fast_power(base:int,power:int,m:int) -> int:
    '''
    return base^power mod m
    '''
    ans = 1
    while power != 0:
        if power & 1:
            ans = ans * base % m
        base = base * base % m
        power >>= 1
    return ans 

def get_all_factors(n:int):
    '''获取n所有的因数'''
    s = set()
    for i in range(1,int(math.sqrt(n) + 1)+1):
        if n % i == 0:
            s.add(i)
            s.add(n // i)
    return s


def factor(n:int):
    '''
    factor(n)
    对n进行素因数分解
    算数基本定理进行分解
    '''
    l = []
    target = int(math.sqrt(n)) + 1
    for i in range(2,target+1):
        while n % i == 0:
            n = n // i
            l.append(i)
    if n != 1:
        l.append(n)
    return dict(Counter(l))

def euler_judge(a:int,p:int):
    '''判断a是不是模p的平方剩余'''
    ans = fast_power(a,(p-1)//2,p)
    return  ans if ans==1 else ans - p

def legendre(a:int,p:int):
    '''计算勒让德符号(a/p)'''
    if a % p == 0:
        return 0
    return euler_judge(a,p)

def theorem_4_3_4(a,p):
    if a == 2:
        return (-1) ** ((p**2-1)/8)
    elif gcd(a,2*p) == 1:
        tmp = 0
        for k in range(1,(p-1)//2 + 1):
            tmp += int(a*k/p)
        return (-1) ** tmp
    else:
        print(f'gcd(a,2p) != 1')


def m2m(m,e,b):
    '''
    return m^e % b
    '''
    result=1
    m1=m
    while(e>=1):
        e1=e%2
        if(e1==1):
            result=(m1*result)%b
        m1=(m1**2)%b
        e=e//2
    return int(result)

# class RSA():
#     @staticmethod
#     def hello_static():
#         print(f'This is {RSA.__name__} static method!')

#     @classmethod
#     def hello_class(cls):
#         print(f'This is {RSA.__name__} class method!')

#     def __init__(self,p=19260817,q=19260817):
#         self.p = p
#         self.q = q
#         self.n = p * q
#         self.phi = (self.p -1) * (self.q -1)
#         self.e = random.randint(2,self.phi)
#         while gcd(self.e,self.phi) != 1:
#             self.e = random.randint(2,self.phi)
#         self.d = get_inverse(self.e,self.phi)
#         self.char_to_index = {}
#         self.index_to_char = {}
#         self.set_char()
    
#     def __func(self,num):
#             if num < 10:
#                 return '0' + str(num)
#             else :
#                 return str(num)
    
#     def set_char(self,char_set = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + 'abcdefghijklmnopqrstuvwxyz' + ' ,.?!' + '0123456789'):
        
        
#         self.char_set = char_set
#         indices = [self.__func(i) for i in range(len(self.char_set))]
#         self.char_to_index = {k:v for k,v in zip(self.char_set,indices)}
#         self.index_to_char = {k:v for v,k in self.char_to_index.items()}


#     def digitalize(self,m):
#         if not isinstance(m,str):
#             m = str(m)
#         ans = []
#         for item in m:
#             ans.append(self.char_to_index.get(item))
#         return ''.join(ans)
    
#     def dedigitalize(self,c):
#         if not isinstance(c,str):
#             c = str(c)
#         ans = []
#         for i in range(0,len(c),2):
#             ans.append(self.index_to_char.get(c[i] + c[i+1]))
#         return ''.join(ans)
        
#     def lock(self,m):
#         if not isinstance(m,int):
#             m = int(m)
#         c = m2m(m,self.e,self.n)
#         return c

#     def unlock(self,c):
#         if not isinstance(c,int):
#             c = int(c)
#         m = m2m(c,self.d,self.n)
#         return m
    
def two2ten(n:str)->int:
    n = list(n)[::-1]
    ans = 0
    for index in range(len(n)):
        ans += int(n[index]) * 2**index
    return ans


def func(n:str):
    k,n = n.split('.')
    base = 10 ** (len(n))
    n = int(n)
    count = 0
    ans = []
    for i in range(40):
        n *= 2
        ans.append(int(n // base))
        count += 1
        if count == 4:
            ans.append(' ')
            count = 0
        n = n % base
        if n == 0:
            break
    weishu = ''.join(str(item) for item in ans)

    return k + '.' + weishu

def jianfa(a:str,b:str):
    '''二进制减法 a需要比b大'''
    A = [0] * len(a)
    B = [0] * len(b)
    for i in range(len(a)):
        A[i] = int(a[i])
    for i in range(len(b)):
        B[i] = int(b[i])
    a = A
    b = B
    a = a[::-1]
    b = b[::-1]
    for index in range(len(b)):
        a[index] -= b[index]
    for index in range(len(a)):
        while a[index] < 0:
            a[index] += 2
            a[index + 1] -= 1
    a = a[::-1]
    return ''.join(str(item) for item in a)

def get_quadratic_residue(m:int):
    '''获取模m的二次剩余'''
    ans = set()
    for x in range(m):
        a = x**2 % m
        if gcd(a, m) == 1:
            ans.add(a)
    return ans

def theorem_4_6_3(a:int,p:int):
    '''课本149页的定理4.6.3'''
    assert p%2==1 and is_prime(p),f'p={p} is not an odd prime number!'
    
    t = int(factor(p-1).get(2,0))
    s = (p-1) // 2**t
    a_inverse = get_inverse(a,p)

    while True:
        n = random.randint(1,p)
        if gcd(n,p) == 1 and legendre(n,p) != 1:
            break
    b = fast_power(n,s,p)
    ans = [0] * t
    ans[-1] = fast_power(a,(s+1)//2,p)
    for index in range(t-2,-1,-1):
        flag = fast_power((a_inverse * ans[index + 1]**2),2**(index),p)
        if flag % p == 1:
            j = 0
        else:
            j = 1
        ans[index] = (ans[index + 1] * b**(j * 2**(t - index - 2))) % p
    return ans[0]

def theorem_4_6_2(a:int,p:int,q:int):
    '''solve this : x^2 mod p*q = a'''
    assert is_prime(p) and (p+1) % 4 == 0,f'You cannot use theorem_4_6_2 because p={p} is not prime like 4k+3'
    assert is_prime(q) and (q+1) % 4 == 0,f'You cannot use theorem_4_6_2 because q={q} is not prime like 4k+3'
    assert legendre(a,p)==1, f'You cannot use theorem_4_6_2 because p={p} does not satisfiy legendre'
    assert legendre(a,q)==1, f'You cannot use theorem_4_6_2 because q={q} does not satisfiy legendre'

    s,t = bezout(q,p)
    s,t = s*q,t*p
    s1 = fast_power(a,(p+1)//4,p)
    t1 = fast_power(a,(q+1)//4,q)
    ans = [0] * 4
    ans[0] = s1*s + t1*t
    ans[1] = -s1*s + t1*t
    ans[2] = s1*s - t1*t
    ans[3] = -s1*s - t1*t
    for index in range(len(ans)):
        ans[index] %= (p*q)
    return tuple(ans)


# @click.command()
# @click.option('--a',type=int)
# @click.option('--m',type=int)
def enumerate_quadratic(a,m):
    '''暴力求解x^2 mod m == a'''
    ans = []
    for x in range(m):
        if x**2 % m == a:
            ans.append(x)
    print(ans)
    return tuple(ans)

class Rabin():
    def __init__(self,p=19260803,q=19260767):
        self.p = p
        self.q = q
        self.n = self.p * self.q 
        self.charset = ' ,.?!' + '0123456789' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + 'abcdefghijklmnopqrstuvwxyz'
        self.char_to_index = {
            char:index + 12345678 for index,char in enumerate(self.charset)
        }
        self.index_to_char = {
            index:char for char,index in self.char_to_index.items()
        }
        self.group_len = 16
    def lock(self,m):
        cypertext = ''
        m = str(m)
        for char in m:
            index = self.char_to_index[char]
            c = str(fast_power(index,2,self.n))
            c = '0'*(self.group_len - len(c)) + c
            cypertext += c
        return cypertext

    def unlock(self,c):
        c = str(c)
        m = ''
        for index in range(0,len(c),self.group_len):
            c_single = int(c[index:index+self.group_len])
            c_num = theorem_4_6_2(c_single,self.p,self.q)
            for num in c_num:
                if num < 1e9:
                    m += self.index_to_char[num]
                    break

        return m


def theorem_4_7_1(p:int):
    '''
    课本P159定理4.7.1
    求解x^2 + y^2 = p
    p = 2 or p = 4k + 1
    '''
    assert p==2 or p%4==1 ,'p != 2 or p is not like 4k + 1'
    x = theorem_4_6_3(-1,p)
    y = 1
    m = (x**2 + y**2)//p
    while m != 1:
        u = x % m 
        v = y % m
        x,y = (u*x + v*y)//m , (u*y - v*x)//m
        m = (x**2 + y**2) //p
    return x,y

def get_exp(a,m):
    '''获得a模m的指数e'''
    assert gcd(a,m) == 1,f'gcd({a},{m})不是1，不满足指数的条件！'
    i = 1
    while fast_power(a,i,m) != 1:
        i += 1
    return i

def is_primitive_root(a:int,m:int):
    '''判断a是不是模m原根'''
    assert gcd(a,m) == 1,f'a = {a} , m = {m} 并不互素，不满足原根或指数的判断条件！'
    flag = True
    phi = euler_function(m)
    e = get_exp(a,m)
    return e == phi
    pass


def get_prime_factors(n):
    '''获取n所有的素因子'''
    l = list(factor(n).keys())
    return l


def get_primitive_root(n):
    '''求n的原根'''
    # assert is_prime(n) and n != 2,f'{n} is not odd prime number!'
    if is_prime(n) and n != 2:
        l = get_prime_factors(n-1)
        l = list(map(lambda x:(n-1)//x,l))
        ans = []
        for g in range(2,n):
            if gcd(g,n) != 1:
                continue
            flag =True
            for index in l:
                if fast_power(g,index,n) == 1:
                    flag = False
                    break
            if flag:
                ans.append(g)
        return ans
    else:
        print(f'm = {n} is not odd prime number, so using enumerate to get all primitive roots of m = {n}')
        ans = []
        for i in range(1,n):
            try:
                if is_primitive_root(i,n):
                    ans.append(i)
            except AssertionError:
                pass
        return ans
    pass
    


class RSA():
    def __init__(self):
        super().__init__()
        self.p = get_large_prime()
        self.q = get_large_prime()
        self.n = self.p * self.q
        self.len = len(str(self.n))
        self.phi = (self.p-1) * (self.q-1)
        # self.bias = 132435343242330
        # self.charset = '0123456789' + 'abcdefghijklmnopqrstuvwxyz' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + ',.?><[]*%_@{#} \\!/:\n()+-=\'"'
        # self.index_to_char = {
        #     index + self.bias :char for index,char in enumerate(self.charset)
        # }
        # self.char_to_index = {
        #     char : index for index,char in self.index_to_char.items()
        # }
        
        self.e = random.randint(2,self.phi)
        while gcd(self.e,self.phi) != 1:
            self.e = random.randint(2,self.phi)
        self.d = get_inverse(self.e,self.phi)

    def lock(self,m):
        m = str(m)
        tmp = []
        for char in m:
            # tmp.append(self.char_to_index[char])
            tmp.append(ord(char))
        c = []
        for char in tmp:
            char = int(char)
            c_tmp = str(fast_power(char,self.e,self.n))
            c_tmp = '0'*(self.len - len(c_tmp)) + c_tmp 
            c.append(c_tmp)
        return ''.join(c)
    
    def unlock(self,c,use_china_res=False):
        if not use_china_res:
            c = str(c)
            m = []
            for i in range(0,len(c),self.len):
                c_char = int(c[i:i+self.len])
                c_char = fast_power(c_char,self.d,self.n)
                # c_char = self.index_to_char[c_char]
                c_char = chr(c_char)
                m.append(str(c_char))
            return ''.join(m)
        else:
            # count = 0
            c = str(c)
            m = []
            for i in range(0,len(c),self.len):
                print(f'Decoding No.{i} char...')
                c_char = int(c[i:i+self.len])
                # c_char = fast_power(c_char,self.d,self.n)
                b1 = fast_power(c_char,self.d,self.p)
                b2 = fast_power(c_char,self.d,self.q)
                c_char = china_res([b1,b2],[self.p,self.q])[0]
                # c_char = self.index_to_char[c_char]
                c_char = chr(c_char)
                m.append(str(c_char))
            return ''.join(m)
    

class DH():
    def __init__(self):
        super().__init__()
        self.p = 353
        gs = get_primitive_root(self.p)
        index = random.randint(0,len(gs))
        self.g = gs[index]
        self.sk = 233
        self.pk = fast_power(self.g,self.sk,self.p)
    def change(self,pk_2):
        self.sk = fast_power(pk_2,self.sk,self.p)


def is_large_prime(n:int,k:int=4):
    '''
    Use Miller-Rabin method to judge whether number n is a large prime number.\n
    k in the safe coefficient and when k = 4(default), the accuracy is bigger than 99.99%.\n
    '''
    if n == 1 or (n & 1) == 0:
        return False
    if n == 2:
        return True
    t = n - 1
    s = 0
    while (t & 1) == 0:
        # t = t >> 1
        t >>= 1
        s += 1

    for _ in range(k):
        b = random.randint(2,n)
        while gcd(b,n) != 1:
            b = random.randint(2,n)
        # r = 0
        index = fast_power(b,t,n)

        # r = 0
        if index == 1 or index == (n-1):
            continue
        
        flag = False
        for r in range(1,s):
            index = (index**2) % n
            if index == 1:
                return False
            if index == (n-1):
                flag = True
            if flag:
                break
        if flag:
            continue
        else:
            return False
    return True

def get_large_prime(low:int=2**100,high:int=2**200,k:int=4):
    '''
    Use Miller-Rabin method to find a large prime number.\n
    Please make sure the difference between low and high is sufficiently big.\n
    k in the safe coefficient and when k = 4(default), the accuracy is bigger than 99.99%.\n
    '''
    while True:
        n = random.randint(low,high)
        if is_large_prime(n,k):
            return n


def lian_fen_shu(x,K:int=10):
    '''
    构造简单的连分数\n
    x是需要计算的数字\n
    K是迭代次数 默认是10\n
    返回一个元组：第1个元素是近似值，第2个元素是连分数，是一个list\n
    '''
    tmps = []
    a = math.floor(x)
    x = x - a
    k = 0
    tmps.append(a)
    while x != 0 and k < K:
        a = math.floor(1 / x)
        x = 1/x - a
        tmps.append(a)
        k += 1
    tmps = tmps[::-1]
    ans = tmps[0]
    for index in range(1,len(tmps)):
        ans = 1/ans + tmps[index]
    # print(tmps[::-1])
    return ans,tmps[::-1]



class Fenshu():
    def __init__(self,num=1,den=1):
        '''num是分子默认为1\nden是分母默认为1\n'''
        super().__init__()
        self.num = int(num)
        self.den = int(den)
    
    def add(self,y):
        num = self.num * y.den + self.den * y.num
        den = self.den * y.den
        tmp = gcd(num,den)
        num //= tmp
        den //= tmp
        return Fenshu(num,den)
    
    def sub(self,y):
        num = self.num * y.den - self.den * y.num
        den = self.den * y.den
        tmp = gcd(num,den)
        num //= tmp
        den //= tmp
        return Fenshu(num,den)
    
    def mul(self,y):
        num = self.num * y.num
        den = self.den * y.den
        tmp = gcd(num,den)
        num //= tmp
        den //= tmp
        return Fenshu(num,den)
    
    def inv(self):
        return Fenshu(self.den,self.num)
    
    def div(self,y):
        return self.mul(y.inv())

    def display(self):
        return self.num,self.den
    
    def xiaoshu(self):
        return self.num / self.den

    def to_lianfenshu(self):
        '''转换成连分数的形式\n如果这个分数是负数，会被转换成正数来进行操作\n'''
        ans = []
        num = abs(self.num)
        den = abs(self.den)
        zhengshu = num // den
        xiaoshu = Fenshu(num - zhengshu * den,den)
        ans.append(zhengshu)
        while xiaoshu.num != 0:
            xiaoshu = xiaoshu.inv()
            zhengshu = xiaoshu.num // xiaoshu.den
            xiaoshu = Fenshu(xiaoshu.num - zhengshu * xiaoshu.den, xiaoshu.den)
            ans.append(zhengshu)
        return ans



def lianfenshu_to_float(X:list):
    '''从一个list的连分数转换成分子/分母\n'''
    X = [Fenshu(x) for x in X]
    ans = X[-1]
    X = X[0:-1][::-1]
    for x in X:
        ans = ans.inv().add(x)
    return ans.display(),ans.xiaoshu()



if __name__ == '__main__':
    # n = 2**257 - 1
    # n = 19260817
    # for p in [89,107]:
    #     n = 2**p - 1
    #     print(f'p = {p} , n = {n} , {is_large_prime(n)}')
    # print(gcd(12,10,6,4,6454))


    # d = DH()
    # d.change(40)
    # a = 61
    # print(get_primitive_root(a))
    # print(is_large_prime(2**67 - 1))
    # n = 2**257 - 1
    # s = 1
    # t = (n-1) // (2**s)
    # print(fast_power(3,t,n))


    # print(lian_fen_shu(math.pi,10) )
    # with open('ans.json','w',encoding='utf8') as f:
    #     ts = [(20210520,113),(210520,191)]
    #     ANS = {}
    #     for index,t in enumerate(ts):
    #         ans = {}
    #         a,b = t
    #         ans['a'] = a
    #         ans['b'] = b
    #         ans['连分数'] = lian_fen_shu(a/b,100)[1]
    #         # print(lian_fen_shu(a/b,100))
    #         # print(bezout(a,b))
    #         ans['a的系数'],ans['b的系数'] = bezout(a,b)
    #         ANS[index] = ans
    #     json.dump(ANS,f,ensure_ascii=False)

    # r = RSA()
    # text = ''
    # with open('out.txt','r',encoding='utf8') as f:
    #     text = f.read()
    # c = r.lock(text)
    # print(c)
    # with open('cipher_text.txt','w+') as f:
    #     f.write(c)
    # print()
    # print(r.unlock(c,False))


    # all = 0
    # true = 0
    # for n in range(100000):
    #     if is_large_prime(n):
    #         all += 1
    #         if is_prime(n):
    #             true += 1
    # print(true / all)

    # a = get_large_prime()
    # print(a,get_prime_factors(a))
    pi = [3,7,15,1,293,10,3,8,2,1,3,11,1,2,1,2,1]
    ans = lianfenshu_to_float(pi)
    num = ans[0][0]
    den = ans[0][1]
    print(ans)
    print(pi)
    print(Fenshu(num,den).to_lianfenshu())
    print(Fenshu(22,7).xiaoshu())


    # a = Fenshu(1,3)
    # b = Fenshu(2,5)
    # c = a.div(b)
    # print(c.display())

    

    # ms = [5,6,7]
    # m = 1
    # for _ in ms:
    #     m *= _
    # Ms = list(map(lambda x:m//x,ms))
    # for Mi,mi in zip(Ms,ms):
    #         print(f'{m}/{mi}={Mi}模{mi}逆元是{get_inverse(Mi,mi)}')


    pass

    # for n in [191,191**2,113,113*9]:
    #     for b in [2,3,5,7]:
    #         print(f'n = {n} , b = {b} , mod = {fast_power(b,n-1,n)}')

# a = [3,1,1,2,3,1,1]
# a = a[::-1]
# ans = a[0]
# for i in range(1,len(a)):
#     ans = 1/ans + a[i]
# print(ans - 7700/2145)


# a = 1
# for i in range(10000):
#     a = 1/a + 1
# print(a - (5**0.5+1)/2)