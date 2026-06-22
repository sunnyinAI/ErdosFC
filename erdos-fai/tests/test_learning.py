"""Learning Layer: trajectories, evaluation, and the optimizer."""

from erdos_fai import Evaluator, Optimizer
from erdos_fai.learning import Trajectory, TrajectoryStep


def _trajectory(final="A clear, grounded final answer with enough words.") -> Trajectory:
    traj = Trajectory(run_id="t1", task="demo task")
    traj.add(TrajectoryStep("s1", "A", "out one", "claude-opus-4-8", 10, 5))
    traj.add(TrajectoryStep("s2", "B", "out two", "claude-opus-4-8", 10, 5, approved=True))
    traj.final_output = final
    return traj


def test_evaluator_scores_all_metrics():
    report = Evaluator().evaluate(_trajectory(), safety_interventions=0)
    names = {m.name for m in report.metrics}
    assert "Task Completion" in names and "Safety" in names
    assert 0.0 <= report.overall <= 1.0
    assert report.overall > 0.8  # healthy trajectory


def test_safety_interventions_lower_score():
    clean = Evaluator().evaluate(_trajectory(), safety_interventions=0).overall
    flagged = Evaluator().evaluate(_trajectory(), safety_interventions=3).overall
    assert flagged < clean


def test_optimizer_refines_low_metrics():
    # Empty final output tanks Faithfulness/Relevancy → lessons produced.
    traj = _trajectory(final="")
    report = Evaluator().evaluate(traj, safety_interventions=2)
    lessons = Optimizer().refine(traj, report)
    assert lessons
    assert all(lesson.score < 0.8 for lesson in lessons)
