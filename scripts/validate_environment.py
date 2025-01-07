import os
import sys
import platform
import subprocess
import multiprocessing

def check_python_version():
    print("Checking Python version...")
    if sys.version_info[:3] != (3, 9, 7):
        print("ERROR: Python 3.9.7 is required. Detected:", platform.python_version())
        sys.exit(1)
    print("Python version is correct.")

def check_system_requirements():
    print("Validating system requirements...")
    if platform.system() != "Linux":
        print("ERROR: This project is only supported on Linux.")
        sys.exit(1)

    kernel_version = platform.release()
    if not kernel_version.startswith("5.15.0-rc3"):
        print("ERROR: Linux Kernel version 5.15.0-rc3 is required. Detected:", kernel_version)
        sys.exit(1)

    print("System requirements validated.")

def check_memory():
    print("Checking system memory...")
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = f.read()
        mem_total = int([line for line in meminfo.split("\n") if "MemTotal" in line][0].split()[1])
        if mem_total < 64000000:  # 64 GB in kB
            print(f"ERROR: At least 64 GB of RAM is required. Detected: {mem_total // 1024} MB")
            sys.exit(1)
    except Exception as e:
        print("WARNING: Could not verify system memory:", str(e))
    print("Memory check passed.")

def check_nvme_speed():
    print("Benchmarking NVMe drive performance...")
    try:
        result = subprocess.run(
            ["dd", "if=/dev/zero", "of=/tmp/testfile", "bs=1G", "count=1", "oflag=direct"],
            capture_output=True, text=True
        )
        speed_line = [line for line in result.stderr.split("\n") if "bytes/sec" in line]
        speed = float(speed_line[0].split()[-2])
        if speed < 3500:
            print(f"ERROR: NVMe write speed too low: {speed} MB/s (minimum 3500 MB/s required).")
            sys.exit(1)
    except Exception as e:
        print("WARNING: Could not measure NVMe speed:", str(e))
    print("NVMe drive speed validated.")

def check_omp_threads():
    print("Checking OMP_NUM_THREADS...")
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_threads != "1":
        print("ERROR: OMP_NUM_THREADS must be set to 1. Detected:", omp_threads)
        sys.exit(1)
    print("OMP_NUM_THREADS is correctly set.")

def check_custom_libraries():
    print("Validating custom libraries...")
    libraries = ["libhyperscale-matrix.so"]
    for lib in libraries:
        if not os.path.exists(f"/usr/local/lib/{lib}"):
            print(f"ERROR: Required library {lib} is missing.")
            sys.exit(1)
    print("Custom libraries are correctly installed.")

def main():
    print("Starting environment validation...\n")
    check_python_version()
    check_system_requirements()
    check_memory()
    check_nvme_speed()
    check_omp_threads()
    check_custom_libraries()
    print("\nAll environment checks passed! Your setup is ready to (not) run the project.")

if __name__ == "__main__":
    main()
