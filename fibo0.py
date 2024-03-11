
from time import time
from concurrent.futures import ProcessPoolExecutor

def fib(n):
    return n if n < 2 else fib(n - 2) + fib(n - 1)

if __name__ == "__main__":
    start_time = time()
    a = []
    for i in range(41):
        a.append(fib(i))
    maxpos = a.index(max(a))
    print(f"fib({maxpos}) = {a[maxpos]}")
    end_time = time()
    print(f"Sem thread: {end_time - start_time:.2f}s")
   