import os

def fork_example():
    print(f"I am the original process. My PID is {os.getpid()}")
    
    pid = os.fork()

    if pid == 0:
        # This code only runs in the child process
        print(f"I am the child! My PID is {os.getpid()} and my parent is {os.getppid()}")
    else:
        # This code only runs in the parent process
        print(f"I am the parent {os.getpid()}! I just created a child with PID {pid}")

if __name__ == "__main__":
    fork_example()