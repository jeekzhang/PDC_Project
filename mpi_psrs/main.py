from mpi4py import MPI
import random
import bisect
import argparse
from itertools import chain

correct_sort = False
mpi_comm = MPI.COMM_WORLD
size = mpi_comm.Get_size()
rank = mpi_comm.Get_rank()

if rank == 0:
    random.seed(3407)
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--length', type=int, required=True, help='Length of the array')
    args = parser.parse_args()
    length = args.length
    random_array = [random.randint(0, 10000000) for _ in range(length)]
    copy_array = [i for i in random_array]
    start_time = MPI.Wtime()
    copy_array.sort()
    end_time = MPI.Wtime()
  
    t_s = end_time - start_time

    start_time = MPI.Wtime()

    # 1. 均匀划分
    partition_size = length // size
    data = [random_array[i : i+partition_size] for i in range(0, (size-1) * partition_size, partition_size)]
    data.append(random_array[(size-1) * partition_size : length])
else:
    data = None

# 2. 局部排序
data = mpi_comm.scatter(data, root=0)
data.sort()

# 3. 选取样本
samples = [data[i] for i in range(0, len(data), max(len(data) // size, 1))][:size]
samples = mpi_comm.gather(samples, root=0)

# 4. 样本排序
if rank == 0:
    samples_array = list(chain(*samples))
    samples_array.sort()

    # 5. 选择主元
    pivots = [samples_array[i] for i in range(size, len(samples_array), size)][:size-1]
else:
    pivots = None

# 6. 主元划分
pivots = mpi_comm.bcast(pivots, root=0)
cut_data = []
start_index = 0
for pivot in pivots:
    index = bisect.bisect_left(data, pivot, start_index)
    cut_data.append(data[start_index:index])
    start_index = index
cut_data.append(data[start_index:])

# 7. 全局交换
recv_data = [cut_data[rank]]

for j in range(size):
    if j != rank:
        # 使用 mpi_comm.sendrecv() 来避免死锁
        recv_data_part = mpi_comm.sendrecv(sendobj=cut_data[j], dest=j, source=j)
        recv_data.append(recv_data_part)

recv_array = list(chain(*recv_data))

# 8. 归并排序
recv_array.sort()
recv_array = mpi_comm.gather(recv_array, root=0)
if rank == 0:
    array = list(chain(*recv_array))
    end_time = MPI.Wtime()
    if array == copy_array:
        correct_sort = True
    t_p = end_time - start_time
    print(t_s, t_p, correct_sort)