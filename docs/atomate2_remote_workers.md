# Remote Cluster: Atomate2 & JobFlow Remote Setup

This document summarizes the configuration and operational steps for running Atomate2 workflows remotely on a Slurm cluster (such as a remote HPC facility) using `jobflow-remote`.

## 1. Project Configuration
The primary configuration for this setup is located in:
`~/.jfremote/<your_jobflow_project>.yaml`

### Key Settings:
- **Worker Type**: `local` (Switched from `remote` to avoid SSH loopback issues on the login nodes).
- **Scheduler**: `slurm`
- **Working Directory**: `/path/to/cluster/scratch/<username>/jobflow_runs`
- **Environment Setup**: Handled via the `pre_run` script in the YAML, which:
  - Loads VASP module (e.g. `module load vasp/6.4.2-cpu`).
  - Activates the `atomate2` conda environment.
  - Sets `export ATOMATE2_CONFIG_FILE=~/.config/atomate2/config.yaml`.

### Project Name Auto-detection
To avoid specifying `project_name` in every command, you can set the `ATOMATE2_REMOTE_PROJECT` environment variable:
```bash
export ATOMATE2_REMOTE_PROJECT=<your_jobflow_project>
```
When this variable is set, Atomate2 MCP tools will automatically use this project if no valid default is found in the `jobflow-remote` configuration.

## 2. Slurm Submission Policies
To comply with cluster node policies, the following Slurm resources are configured:
- **QoS**: `regular` (Or the appropriate cluster QoS).
- **Constraint**: `cpu` (Or cluster-specific node architecture constraint).
- **Time**: "24:00:00" (Default).
- **Note**: Explicit `partition` naming may be omitted if your cluster maps queues based on QoS.

## 3. Database & Stores
The setup connects to a MongoDB instance (`<mongodb_host>:27017`).

- **Queue Store**: Stores job states and metadata (`agent_db.queue`).
- **Job Store (Outputs)**: Stores calculation results (`agent_db.outputs`).
- **Additional Stores**:
  - `data`: Required for large datasets/specific Atomate2 outputs (`agent_db.data`).

### Full-split Configuration: SSH Tunnel for Local Access

If you are querying Atomate2 data from your **local laptop/workstation** (not directly on the cluster), you need an SSH tunnel to access the cluster MongoDB.

**Configuration Overview:**
- **`jobstore`**: Uses `localhost:27017` (for local queries via SSH tunnel)
- **`remote_jobstore`**: Uses `<mongodb_host>:27017` (for remote cluster - direct access, no tunnel needed)

#### Automatic SSH Tunnel Setup

Add this to your `~/.bashrc` on your **local machine**:

```bash
# Cluster MongoDB Auto-tunnel with Key Expiration Check
cluster_tunnel_start() {
    local CLUSTER_KEY="$HOME/.ssh/<cluster_key>"

    # Check if key exists
    if [[ ! -f "$CLUSTER_KEY" ]]; then
        echo "⚠️  SSH key not found. Run your cluster auth command (e.g. sshproxy -u <username>)" >&2
        return 1
    fi

    # Check key age (< 24 hours)
    local key_age_seconds=$(($(date +%s) - $(stat -c %Y "$CLUSTER_KEY" 2>/dev/null || stat -f %m "$CLUSTER_KEY")))
    local key_age_hours=$((key_age_seconds / 3600))

    if [[ $key_age_hours -ge 24 ]]; then
        echo "⚠️  SSH key expired ($key_age_hours hours old). Please refresh." >&2
        return 1
    fi

    # Check if tunnel already running
    if pgrep -f "ssh.*<mongodb_host>.*27017" > /dev/null 2>&1; then
        return 0
    fi

    # Create tunnel
    ssh -f -N -L 27017:<mongodb_host>:27017 -i "$CLUSTER_KEY" <username>@<login_hostname> 2>/dev/null
}

# Auto-start tunnel on login
cluster_tunnel_start

# Helpful aliases
alias cluster-refresh='sshproxy -u <username> && pkill -f "ssh.*<mongodb_host>.*27017"; cluster_tunnel_start'
alias cluster-status='pgrep -f "ssh.*<mongodb_host>.*27017" > /dev/null && echo "✅ Tunnel running" || echo "❌ Tunnel not running"'
```

Replace `<username>`, `<login_hostname>`, and `<cluster_key>` with your cluster credentials.

**What this does:**
- ✅ Auto-creates SSH tunnel on login if SSH key is fresh (< 24 hours)
- ⚠️ Warns if key is expired (won't create broken tunnel)
- ✅ Prevents duplicate tunnels
- ✅ Provides easy refresh: `cluster-refresh`

**Daily workflow:**
1. Login to your laptop - tunnel auto-starts if key is valid
2. When key expires (~24h), you'll see a warning
3. Refresh your credentials (e.g. `sshproxy -u <username>`)
4. Tunnel auto-restarts on next login or manual `cluster-refresh`

**Check tunnel status:**
```bash
ps aux | grep "<mongodb_host>.*27017"
```

## 4. Runner Operations
The JobFlow Remote Runner operates as a background daemon managed by `supervisord`.

### Common Commands:
- **Start Runner**:
  ```bash
  export PATH=/path/to/conda/envs/atomate2/bin:$PATH
  jf -p <your_jobflow_project> runner start --log-level info
  ```
- **Stop Runner**:
  ```bash
  jf -p <your_jobflow_project> runner stop
  ```
- **Check Status**:
  ```bash
  jf -p <your_jobflow_project> runner status
  ```
- **Reset Daemon**: (Use if the daemon is stuck or reports running on another machine)
  ```bash
  jf -p <your_jobflow_project> runner reset
  ```

## 5. Remote Workflow: How to use it
Once this setup is complete, you can submit jobs from your **local machine** and they will be executed on **a remote cluster** without you needing to be logged in.

### Step-by-Step Execution:
1. **Submit locally**: Run your Atomate2 flow on your local machine using the `<your_jobflow_project>` project target. This uploads the jobs to the MongoDB queue.
2. **Daemon takes over**: The `jobflow-remote` runner daemon on the remote cluster periodically checks the MongoDB queue for `READY` jobs.
3. **Execution**: The daemon submits the jobs to Slurm, monitors them, and downloads results back to the `outputs` and `data` collections in MongoDB upon completion.

### Common Questions:
- **Do I need to start the runner every time?** No. Once started with `runner start`, it runs as a background process using `supervisord`. It will survive your logout.
- **Will jobs be submitted if I log out?** Yes. As long as the cluster login node is up and the daemon hasn't been stopped, it will continue to process any new jobs you submit from your local machine.
- **How do I know if it's still alive?** Log in to the remote cluster and run `jf -p <your_jobflow_project> runner status`. If it says `Daemon status: running`, you are good to go.
- **What if the cluster reboots?** If the login node is rebooted, the daemon will stop. You will need to log in and run the `runner start` command again.

## 6. Job Management
- **List Jobs**: `jf -p <your_jobflow_project> job list`
- **Check Slurm Queue**: `squeue -u <username>`
- **Runner Logs**: `~/.jfremote/<your_jobflow_project>/log/runner.log`

---
*Setup completed/verified on 2026-01-15.*
*SSH tunnel auto-configuration added on 2026-01-22.*
