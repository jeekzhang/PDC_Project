import subprocess
from tqdm import tqdm

def run_script(procs, length):
    command = ["mpiexec", "-n", f"{procs}", "python", "main.py", "-l", f"{length}"]
    result = subprocess.run(command, capture_output=True, text=True)
    output_lines = result.stdout.strip().split('\n')
    t_s, t_p, correct_sort_str = output_lines[0].split()
    t_s = float(t_s)
    t_p = float(t_p)
    correct_sort = bool(correct_sort_str)
    return t_s, t_p, correct_sort

num_runs = 100
length_list = [1000, 5000, 10000, 100000, 1000000, 10000000]
proc_list = [1, 2, 4, 8, 16, 32]

for length in length_list:
    for procs in proc_list:
        Ts = 0.
        Tp = 0.
        Correct = True
        progress_bar = tqdm(range(num_runs), desc=f"{procs} procs, {length} length")
        for _ in progress_bar:
            t_s, t_p, correct_sort = run_script(procs, length)
            Ts += t_s
            Tp += t_p
            Correct = Correct and correct_sort
        progress_bar.set_postfix(Speedup=Ts/Tp, Status="AC" if Correct else "WA")
        print(procs, length, ":", Ts/Tp, "AC" if Correct else "WA")
