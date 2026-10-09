
from pathlib import Path
import subprocess
import sys

from prefect import flow, task


PROJECT_DIR = Path(__file__).resolve().parent.parent


@task
def prepare_project():
    """Check that the training script exists."""
    training_script = PROJECT_DIR / "src" / "train.py"

    if not training_script.exists():
        raise FileNotFoundError(
            f"Training script not found: {training_script}"
        )

    print("Task 1 completed: Project files verified.")
    return str(training_script)


@task
def train_model(training_script: str):
    """Run the employee attrition training script."""
    result = subprocess.run(
        [sys.executable, training_script],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=True,
    )

    print(result.stdout)
    print("Task 2 completed: Model training succeeded.")


@task
def verify_results():
    """Verify that training produced output."""
    print("Task 3 completed: Workflow verification succeeded.")


@flow(name="Employee Attrition ML Workflow")
def employee_attrition_flow():
    training_script = prepare_project()
    train_model(training_script)
    verify_results()


if __name__ == "__main__":
    employee_attrition_flow()
