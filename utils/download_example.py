import os
from huggingface_hub import hf_hub_download

local_dir = "example"
os.makedirs(local_dir, exist_ok=True)

file_path = hf_hub_download(
    repo_id="allenai/objaverse",
    filename="glbs/000-138/3b61335c2a004a9ea31c8dab59471222.glb",
    repo_type="dataset",
    local_dir=local_dir
)
print(f"Downloaded file at: {file_path}")
