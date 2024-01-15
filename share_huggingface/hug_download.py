from huggingface_hub import hf_hub_download
file_name = input("파일 이름을 입력하세요: ")
hf_hub_download(repo_id="exena/boostcamp-klue", filename=file_name)