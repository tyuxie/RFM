import os
import socket
import subprocess
import sys
import time

shared_dir = ".torch_dist"


def machine_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.9.8", 101))
    return s.getsockname()[0]


def send_success(file_path):
    os.system(f"touch {file_path}")


def get_success(file_path):
    if os.path.exists(file_path):
        return True
    return False


def get_master_addr(exp_id, machine_rank, num_machine):
    shared_init_file = os.path.join(shared_dir, exp_id + ".txt")
    master_shared_success_file = os.path.join(shared_dir, f"{exp_id}_0_success")
    shared_success_file = os.path.join(shared_dir, f"{exp_id}_{machine_rank}_success")
    if machine_rank == 0:
        os.makedirs(os.path.dirname(shared_init_file), exist_ok=True)
        master_addr = machine_ip()
        with open(shared_init_file, "w") as f:
            f.write(master_addr)
        # wait a second to avoid network io speed too slow
        time.sleep(1)
        send_success(master_shared_success_file)
    else:
        while not get_success(master_shared_success_file):
            time.sleep(1)
        time.sleep(1)
        with open(shared_init_file, "r") as f:
            master_addr = f.readlines()[0].strip()

        send_success(shared_success_file)
    # check and make sure that all node is ready, otherwise sleep
    while not all(
        [
            get_success(os.path.join(shared_dir, f"{exp_id}_{i}_success"))
            for i in range(num_machine)
        ]
    ):
        print(f"{machine_rank} is sleeping", flush=True)
        time.sleep(1)
    return master_addr


def safe_set_env(key, value):
    if key in os.environ:
        print(
            f"{key} exists in environment, will not change it from {os.environ[key]}to {value}",
            flush=True,
        )
    else:
        print(f"set environment variable {key} to {value}", flush=True)
        os.environ[key] = value

def configure_nccl():
    safe_set_env("NCCL_LAUNCH_MODE", "PARALLEL")
    safe_set_env(
        "NCCL_IB_HCA",
        subprocess.getoutput(
            "cd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
            "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
            "| grep v >/dev/null && echo $i ; done; > /dev/null"
        ),
    )
    safe_set_env("NCCL_IB_GID_INDEX", "3")
    safe_set_env("NCCL_IB_TC", "106")
    safe_set_env("NCCL_DEBUG", "INFO")
    safe_set_env("NCCL_IB_DISABLE", "1")


def configure_dist_env(exp_id):
    machine_rank = int(os.environ.get("RLAUNCH_REPLICA", "0"))
    num_machine = int(os.environ.get("RLAUNCH_REPLICA_TOTAL"))
    master_addr = get_master_addr(exp_id, machine_rank, num_machine)
    master_port = "10666"

    safe_set_env("NNODES", str(num_machine))
    safe_set_env("NODE_RANK", str(machine_rank))
    safe_set_env("MASTER_ADDR", master_addr)
    safe_set_env("MASTER_PORT", master_port)


def main():

    exp_id = sys.argv[1]
    command = sys.argv[2:]
    safe_set_env("OMP_NUM_THREADS", "1")
    # safe_set_env("OMP_NUM_THREADS", "4")
    configure_dist_env(exp_id)
    configure_nccl()
    command = " ".join(command).replace("%", "$")
    print(command)
    print("Excute: ", command, flush=True)
    os.system(command)
    # clean shared file
    pattern = os.path.join(shared_dir, exp_id)
    os.system(f"rm {pattern}*")


if __name__ == "__main__":
    main()
