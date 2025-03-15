import torch
import time
from multiprocessing import Process

def reserve_gpu(device, tensor_size=1024, compute_interval=10):
    torch.cuda.set_device(device)
    reserved_tensor = torch.empty((tensor_size,), device=device)
    print(f"Device {device}: Reserved {reserved_tensor.numel() * reserved_tensor.element_size() / (1024**2):.2f} MB of GPU memory.")
    while True:
        a = torch.randn((16, 16), device=device)
        b = torch.randn((16, 16), device=device)
        c = torch.mm(a, b)
        time.sleep(compute_interval)

def main():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs detected!")
        return

    processes = []
    for i in range(num_gpus):
        p = Process(target=reserve_gpu, args=(i,))
        p.start()
        processes.append(p)
        print(f"Started reservation process for GPU {i}")

    print(f"Reserved resources on {num_gpus} GPU(s). Press Ctrl+C to exit.")
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Exiting GPU reservation script.")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=0,1,2,3 python reserve_gpus.py
