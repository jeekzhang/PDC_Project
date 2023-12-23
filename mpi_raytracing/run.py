import subprocess
from tqdm import tqdm


def run_parallel_script(procs):
    command = ["mpiexec", "-n", f"{procs}", "python", "parallel.py"]
    result = subprocess.run(command, capture_output=True, text=True)
    output_lines = result.stdout.strip().split("\n")
    t_p = float(output_lines[0].split()[0])
    return t_p


def run_serial_script():
    command = ["python", "serial.py"]
    result = subprocess.run(command, capture_output=True, text=True)
    output_lines = result.stdout.strip().split("\n")
    t_s = float(output_lines[0].split()[0])
    return t_s


num_runs = 10
proc_list = [1, 2, 4, 8, 16, 32]

Ts = 0.0
for _ in tqdm(range(num_runs)):
    Ts += run_serial_script()

for procs in proc_list:
    Tp = 0.0
    progress_bar = tqdm(range(num_runs), desc=f"{procs} procs")
    for _ in progress_bar:
        t_p = run_parallel_script(procs)
        Tp += t_p
    progress_bar.set_postfix(Speedup=Ts / Tp)
    print(procs, " Speedup:", Ts / Tp)
