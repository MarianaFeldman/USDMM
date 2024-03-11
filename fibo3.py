from concurrent.futures import ProcessPoolExecutor
from time import time

def fib(n):
    return n if n < 2 else fib(n - 2) + fib(n - 1)

if __name__ == "__main__":
    start_time = time()
    #for _ in range(multiprocessing.cpu_count()):
    #    multiprocessing.Process(target=fib, args=(40,)).start()
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(fib, range(41)))
        #for i, result in enumerate(results):
            #print(f"fib({i}) = {result}")
    #results = [future.result() for future in futures]
    maxpos = results.index(max(results))
    print(f"fib({maxpos}) = {results[maxpos]}")

    print(f"ProcessPoolExecutor: {(time() - start_time):.2f}s")
    #for i, result in enumerate(results):
    #        print(f"fib({i}) = {result}")
    #print(results)