import os
from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")

# Change this filename if your final CSV has a different name
CSV_PATH = "/kaggle/working/drug_reviews_imputed_rf.csv"
REPO_ID = "yogeshagowdaiiitdwd/drug-reviews-final-dataset"


def push_final_dataset(csv_path, repo_id, token):
    api = HfApi(token=token)
    who = api.whoami()
    print("Token belongs to:", who["name"])

    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=csv_path,
        path_in_repo=os.path.basename(csv_path),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Pushed {csv_path} to https://huggingface.co/datasets/{repo_id}")


push_final_dataset(CSV_PATH, REPO_ID, hf_token)
