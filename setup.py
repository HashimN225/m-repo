from setuptools import setup, find_packages

setup(
    name="employee_attrition_project",
    version="0.1.0",
    packages=find_packages(where="src"),
    install_requires=["pandas", "mlflow", "boto3"],
    package_dir={"": "src"}
)
