from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="/data/ephemeral/optimizer.pt",
    path_in_repo="optimizer.pt",
    repo_id="exena/boostcamp-klue",
)