import multiprocessing
from time import time

def fib(n):
    return n if n < 2 else fib(n - 2) + fib(n - 1)

if __name__ == "__main__":
    start_time = time()
    #for _ in range(multiprocessing.cpu_count()):
    #    multiprocessing.Process(target=fib, args=(40,)).start()
    
    with multiprocessing.Pool() as pool:
        results = pool.map(fib, range(40))
        #for i, result in enumerate(results):
            #print(f"fib({i}) = {result}")
    print(f"Multiprocessing: {time() - start_time:.2f}s")
    for i, result in enumerate(results):
            print(f"fib({i}) = {result}")