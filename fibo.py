import os
import threading
from time import time

def fib(n):
    return n if n < 2 else fib(n - 2) + fib(n - 1)

start_time = time()
for _ in range(os.cpu_count()):
    threading.Thread(target=fib, args=(40,)).start()
    #t.run()
print(f"Threading: {time() - start_time:.2f}s")