"""vLLM service management for SIMAX."""

import subprocess
import time
import requests
import os

import sys
from src.utils.io import setup_logger

logger = setup_logger("services")


def check_vllm_status(base_url: str = "http://localhost:8000") -> bool:
    """Check if vLLM is running."""
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            logger.info("vLLM service is running.")
            return True
    except requests.exceptions.RequestException:
        pass
    return False


def start_vllm(config: dict) -> subprocess.Popen:
    """Start the vLLM server."""
    gpt_config = config.get("models", {}).get("gpt", {})
    base_url = gpt_config.get("base_url", "http://localhost:8000/v1")
    
    host, port = _parse_base_url(base_url)
    
    if check_vllm_status(f"http://{host}:{port}"):
        return None
    
    os.makedirs("logs", exist_ok=True)
    
    gpt_model_path = config.get("paths", {}).get("gpt_model_path")
    if not gpt_model_path:
        raise ValueError("Missing config.paths.gpt_model_path")
    
    tensor_parallel_size = str(gpt_config.get("tensor_parallel_size", 1))
    gpu_memory_utilization = str(gpt_config.get("gpu_memory_utilization", 0.9))
    max_model_len = str(gpt_config.get("max_model_len", 4096))
    model_name = gpt_config.get("model_name", "gpt-oss-20b")
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", gpt_model_path,
        "--served-model-name", model_name,
        "--tensor-parallel-size", tensor_parallel_size,
        "--gpu-memory-utilization", gpu_memory_utilization,
        "--max-model-len", max_model_len,
        "--host", host,
        "--port", str(port),
        "--trust-remote-code",
    ]
    
    logger.info(f"Starting vLLM with command: {' '.join(cmd)}")
    log_filename = f"logs/vllm_server_{port}.log"
    log_file = open(log_filename, "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    
    logger.info(f"Waiting for vLLM to start (log: {log_filename})...")
    for _ in range(60):
        if check_vllm_status(f"http://{host}:{port}"):
            logger.info("vLLM started successfully.")
            return process
        time.sleep(10)
    
    logger.error("vLLM failed to start within timeout.")
    raise RuntimeError("vLLM failed to start")


def stop_vllm(port: int = 8000):
    """Stop the vLLM server on the specified port."""
    logger.info(f"Stopping vLLM service on port {port}...")
    try:
        # Search for process with specific port argument
        cmd = f"pgrep -f 'vllm.entrypoints.openai.api_server.*--port {port}'"
        result = subprocess.run(
            cmd,
            shell=True, capture_output=True, text=True
        )
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    logger.info(f"Killing vLLM process {pid}")
                    subprocess.run(f"kill {pid}", shell=True)
            logger.info(f"vLLM service on port {port} stopped.")
        else:
            logger.info(f"No running vLLM service found on port {port}.")
    except Exception as e:
        logger.error(f"Failed to stop vLLM: {e}")


def ensure_services(config: dict, step: str):
    """Ensure necessary services are running for the given step."""
    if step in ["role", "dialogue"]:
        gpt_config = config.get("models", {}).get("gpt", {})
        base_url = gpt_config.get("base_url", "http://localhost:8000/v1")
        host, port = _parse_base_url(base_url)
        
        if not check_vllm_status(f"http://{host}:{port}"):
            start_vllm(config)


def _parse_base_url(base_url: str) -> tuple:
    """Parse base_url to extract host and port."""
    if "://" in base_url:
        base_url = base_url.split("://")[1]
    if "/v1" in base_url:
        base_url = base_url.replace("/v1", "")
    
    parts = base_url.split(":")
    host = parts[0] if len(parts) > 0 else "localhost"
    port = int(parts[1]) if len(parts) > 1 else 8000
    return host, port


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "stop_vllm":
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
            stop_vllm(port)
        else:
            print(f"Unknown command: {sys.argv[1]}")
    else:
        print("Usage: python -m src.services [stop_vllm] [port]")
