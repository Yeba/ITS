import numpy as np

def t_len(t: np.ndarray) -> float: return sum([np.linalg.norm(t[i, :2] - t[i + 1, :2]) for i in range(len(t) - 1)])

def t2ps_its(T: np.ndarray, h: int):
    l = t_len(T)
    vec = np.array([T[0]] * h)
    vec[-1] = T[-1]
    if l < 1e-12: return vec
    e = l / (h - 1)
    a, b = 0, 0
    j = 0
    for i in range(1, len(T)):
        p1, p2 = T[i - 1], T[i]
        s = np.linalg.norm(p1[:2] - p2[:2], 2)
        if s<1e-12:continue
        b += s
        while b >= a:
            if j >= h: break
            vec[j] = (p2 - p1) * (1 - (b - a) / s) + p1
            j += 1
            a += e
    return vec

def ITS(T1: np.ndarray, T2: np.ndarray, h):return np.linalg.norm(t2ps_its(T1, h)- t2ps_its(T2, h)) / np.sqrt(h)
