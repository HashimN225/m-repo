from setuptools import setup, find_packages

setup(
    name="employee_attrition_project",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "": ["datasets/*.csv"],
    },
    install_requires=["pandas", "mlflow", "boto3"],
)
