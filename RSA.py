import random

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



def gcd(*a):
    '''
    求多个整数的最大公因数
    gcd(12,32,45,64)
    '''
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
        t = t >> 1
        s += 1

    for _ in range(k):
        b = random.randint(2,n)
        while gcd(b,n) != 1:
            b = random.randint(2,n)
        index = fast_power(b,t,n)

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


def get_inverse(a,m):
    '''求解a模m的逆元'''
    tmp = bezout(a,m)[0]
    while tmp <= 0:
        tmp += m
    return tmp

class RSA():
    def __init__(self):
        super().__init__()
        # 随机生成p q两个大素数
        self.p = get_large_prime()
        self.q = get_large_prime()
        self.n = self.p * self.q
        self.len = len(str(self.n))

        # 计算欧拉函数
        self.phi = (self.p-1) * (self.q-1)
        self.e = random.randint(2,self.phi)
        while gcd(self.e,self.phi) != 1:
            self.e = random.randint(2,self.phi)
        self.d = get_inverse(self.e,self.phi)

    def lock(self,m):
        m = str(m)
        tmp = []
        for char in m:
            tmp.append(ord(char))
        c = []
        for char in tmp:
            char = int(char)
            c_tmp = str(fast_power(char,self.e,self.n))
            c_tmp = '0'*(self.len - len(c_tmp)) + c_tmp 
            c.append(c_tmp)
        return ''.join(c)
    
    def unlock(self,c):
        c = str(c)
        m = []
        for i in range(0,len(c),self.len):
            c_char = int(c[i:i+self.len])
            c_char = fast_power(c_char,self.d,self.n)
            c_char = chr(c_char)
            m.append(str(c_char))
        return ''.join(m)



if __name__ == '__main__':

    r = RSA()
    text = ''
    # 需要加密的明文放在这里
    with open('plain_text.txt','r',encoding='utf8') as f:
        text = f.read()
    c = r.lock(text)
    print(c)
    # 加密后的密文放在这里
    with open('cipher_text.txt','w+') as f:
        f.write(c)
    print()
    print(r.unlock(c))
    # print(r.q)
    # print(r.e)
    # print(r.p)
    # print(r.d)
