import numpy as np
# 感知机 与门
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    theta = -0.7
    res = w @ x + theta
    return 1 if res > 0 else 0


# 测试与门
print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))

# 感知机 与非门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    theta = 0.7
    res = w @ x + theta
    return 1 if res > 0 else 0


# 测试与非门
print('\n')
print(NAND(0, 0))
print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))


# 感知机 或门
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    theta = -0.4
    res = w @ x + theta
    return 1 if res > 0 else 0

# 测试 或门
print('\n')
print(OR(0, 0))
print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))


# 感知机 异或门
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)

# 测试 异或门
print('\n')
print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))



