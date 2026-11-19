from dataclasses import fields

from geoinvlab.project import Project


def test_project_default_values():
    project = Project()

    assert project.name == "Untitled"
    assert project.method is None
    assert project.dataset is None
    assert project.mesh is None
    assert project.model is None
    assert project.operator is None
    assert project.inverse_problem is None
    assert project.result is None


def test_project_custom_name():
    project = Project(name="Test Project")

    assert project.name == "Test Project"


def test_project_is_dataclass():
    project_fields = {field.name for field in fields(Project)}

    assert project_fields == {
        "name",
        "method",
        "dataset",
        "mesh",
        "model",
        "operator",
        "inverse_problem",
        "result",
    }


def test_project_accepts_workflow_objects():
    class Dummy:
        pass

    method = Dummy()
    dataset = Dummy()
    mesh = Dummy()
    model = Dummy()
    operator = Dummy()
    inverse_problem = Dummy()
    result = Dummy()

    project = Project(
        name="Test",
        method=method,
        dataset=dataset,
        mesh=mesh,
        model=model,
        operator=operator,
        inverse_problem=inverse_problem,
        result=result,
    )

    assert project.method is method
    assert project.dataset is dataset
    assert project.mesh is mesh
    assert project.model is model
    assert project.operator is operator
    assert project.inverse_problem is inverse_problem
    assert project.result is result

