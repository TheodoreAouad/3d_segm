import subprocess


requirements_file = "./requirements2.txt"

with open(requirements_file, "r") as f:
    requirements = f.read().splitlines()

failed = []

for requirement in requirements:
    try:
        subprocess.run(["pip", "install", requirement])
    except Exception as e:
        print(f"Failed to install {requirement}: {e}")
        failed.append(requirement)

print(f"Failed to install: {failed}")