import ray


def compute_resource():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    # This following section computes 
    cluster_resources = ray.available_resources()
    total_cpus,total_gpus = int(cluster_resources.get("CPU", 1)),int(cluster_resources.get("GPU", 0))
    # 2. Reserve resources for your primary Learner (Optimization engine)
    # Typically, assign 1 Learner per GPU. If no GPUs exist, the local driver handles it.
    num_learners = total_gpus if total_gpus > 0 else 0
    num_gpus_per_learner = 1 if total_gpus > 0 else 0

    # 3. Calculate remaining CPUs for the parallel Environment Workers (EnvRunners)
    # Leave 2 CPUs free for system overhead (the Ray driver and Redis/storage logging)
    available_worker_cpus = max(1, total_cpus - (num_learners * 1) - 2)

    return [available_worker_cpus, num_learners, num_gpus_per_learner,total_gpus]