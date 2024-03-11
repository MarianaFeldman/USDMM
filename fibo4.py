from concurrent.futures import ThreadPoolExecutor
from time import time

def fib(n):
    return n if n < 2 else fib(n - 2) + fib(n - 1)

if __name__ == "__main__":
    start_time = time()
    #for _ in range(multiprocessing.cpu_count()):
    #    multiprocessing.Process(target=fib, args=(40,)).start()
    a = []
    with ThreadPoolExecutor() as executor:
        futures= [executor.submit(fib, i) for i in range(41)]
        #for i, result in enumerate(results):
            #print(f"fib({i}) = {result}")
    results = [future.result() for future in futures]
    maxpos = results.index(max(results))
    print(f"fib({maxpos}) = {results[maxpos]}")
    print(f"ThreadPoolExecutor: {time() - start_time:.2f}s")
    