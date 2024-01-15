from huggingface_hub import upload_file
file_path = input("파일 경로 혹은 이름을 입력하세요: ")
upload_file(
    path_or_fileobj=file_path,
    path_in_repo=file_path.split("/")[-1],
    repo_id="exena/boostcamp-klue",
)