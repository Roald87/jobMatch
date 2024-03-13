import argparse
import os
import subprocess


def create_new_job(company, position):
    # Replace spaces with dashes
    company = company.replace(' ', '-')
    position = position.replace(' ', '-')

    folder_name = f"{company}-{position}"
    cv_file = f"{folder_name}-cv.txt"
    job_file = f"{folder_name}-job.txt"
    folder_path = os.path.join("jobs", folder_name)

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Create the CV and job files
    cv_path = os.path.join(folder_path, cv_file)
    job_path = os.path.join(folder_path, job_file)
    open(cv_path, 'a').close()
    open(job_path, 'a').close()

    # Initialize a git repository
    subprocess.run(["git", "init", folder_path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new job folder with CV and job description files")
    parser.add_argument("--company", required=True, help="The name of the company")
    parser.add_argument("--position", required=True, help="The position being applied for")
    args = parser.parse_args()

    create_new_job(args.company, args.position)
