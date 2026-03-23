import paramiko
import os
import random

# Constants - loaded from environment variables
hostname = os.environ.get("SAGA_HOSTNAME", "login-4.saga.sigma2.no")
username = os.environ.get("SAGA_USERNAME", "")
private_key_path = os.environ.get("SAGA_KEY_PATH", os.path.expanduser("~/.ssh/id_rsa"))
passphrase = os.environ.get("SAGA_PASSPHRASE", "")
ignorelist = ["node-01", "node-02", "node-03", "node-04", "node-gpu-1", "node-gpu-2"]

# Setup SSH connection with paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

def connect_ssh():
    private_key = paramiko.RSAKey.from_private_key_file(private_key_path, password=passphrase)
    client.connect(hostname=hostname, username=username, pkey=private_key)

def get_available_idle_nodes(ignorelist=ignorelist, enforced_node=""):
    """
    Fetches available idle nodes from the cluster and sorts by idle cores.
    """
    command = 'sinfo --noheader --Node -o "%N %C" -t idle,mix'
    if enforced_node:
        command += f" | grep {enforced_node}"
    
    stdin, stdout, stderr = client.exec_command(command)
    idle_nodes = stdout.read().decode().splitlines()
    idle_nodes = [(node.split()[0], int(node.split()[1].split('/')[1])) for node in idle_nodes]
    processed_nodes = [node[0] for node in idle_nodes if node[0] not in ignorelist]
    random.shuffle(processed_nodes)
    return processed_nodes[:4]

def run_slurm(node_name, account, target_feature, outputlogs_path, py_source, source_path, args, CPU_PER_TASK=2):
    """
    Submit a Slurm job to a specific node remotely via SSH.
    """
    runner_sh_output_path = os.path.join(outputlogs_path, "slurm-logs")
    job_command = (
        f'sbatch --job-name={target_feature} --account={account} --output={runner_sh_output_path}/%j.log '
        f'--error={runner_sh_output_path}/%j.log --time=4-10:00:00 --nodes=1 '
        f'--ntasks=1 --cpus-per-task={CPU_PER_TASK} --mem=3G '
        f'--nodelist={node_name} '
        f'--wrap="module load Python/3.10.4-GCCcore-11.3.0 && source {py_source} && cd {source_path} && python scripts/lrbased_teleconnection/main.py '
        f'--target_feature={args.target_feature} --modelname={args.modelname} --splitsize={args.splitsize} '
        f'--data_file={args.data_file} --max_allowed_features={args.max_allowed_features} '
        f'{("--with_mean_feature" if args.with_mean_feature else "")}"'
    )

    print(job_command)
    stdin, stdout, stderr = client.exec_command(job_command)
    print("Job submission output:", stdout.read().decode())

def run_optimization(args, execution_mode, jupyter_single_run):
    """
    Main function to run optimizations based on the selected execution mode.
    """
    connect_ssh()

    if execution_mode == "Single Run":
        print(f"Running optimization in {execution_mode} mode...")
        file_output_path = run_single_optimization(args, jupyter_single_run)
        client.close()
        return True

    elif execution_mode in ["Parallel Run", "Cluster Run"]:
        node_name = os.getenv('SLURMD_NODENAME') if execution_mode == "Parallel Run" else ""
        available_nodes = get_available_idle_nodes(enforced_node=node_name)
        
        if available_nodes:
            node = available_nodes[0]
            run_slurm(
                node_name=node,
                account=args.slurmaccount,
                target_feature=args.target_feature,
                outputlogs_path=args.outputlogs_path,
                py_source=args.py_source,
                source_path=args.source_path,
                args=args,
                CPU_PER_TASK=2
            )
            client.close()
            return True
        else:
            print("No available nodes")
            client.close()
            return False
