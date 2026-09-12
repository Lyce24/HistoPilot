"""Resolve batch predictor intent without changing historical frozen meaning."""

from histopilot.schemas.predictor_policy import ExperimentPredictorPolicy
from histopilot.storage.project_lock import StorageError


def resolve_batch_policy(spec, fallback=None):
    return ExperimentPredictorPolicy.model_validate(
        spec.get("predictorPolicy") or fallback or {}
    ).model_dump()


def has_predictor_intent(submission):
    return bool(submission) and (
        "predictorPolicyVersion" in submission
        or "predictorPolicies" in submission
        or bool(submission.get("predictorPolicy"))
    )


def submission_policies(submission):
    if "predictorPolicyVersion" in submission and (
        type(submission["predictorPolicyVersion"]) is not int
        or submission["predictorPolicyVersion"] != 2
        or "predictorPolicies" not in submission
    ):
        raise StorageError(
            "The submitted per-batch predictor policy version or choices changed.",
            "EXPERIMENT_PREDICTOR_PLAN_CHANGED",
            409,
        )
    if "predictorPolicies" in submission:
        policies = submission["predictorPolicies"]
        if not isinstance(policies, dict) or set(policies) != set(submission["batchIds"]):
            raise StorageError(
                "The submitted per-batch predictor choices are incomplete or changed.",
                "EXPERIMENT_PREDICTOR_PLAN_CHANGED",
                409,
            )
        try:
            return {
                key: ExperimentPredictorPolicy.model_validate(value).model_dump()
                for key, value in policies.items()
            }
        except ValueError as error:
            raise StorageError(
                "A submitted batch predictor choice is invalid or changed.",
                "EXPERIMENT_PREDICTOR_PLAN_CHANGED",
                409,
            ) from error
    policy = submission.get("predictorPolicy")
    return (
        {
            batch: ExperimentPredictorPolicy.model_validate(policy).model_dump()
            for batch in submission.get("batchIds", [])
        }
        if policy
        else {}
    )


def predictor_work_expected(submission):
    return has_predictor_intent(submission) and (
        any(value["method"] != "skip" for value in submission_policies(submission).values())
        or any(
            resolve_batch_policy(row["spec"], submission.get("predictorPolicy"))["method"] != "skip"
            for row in submission.get("publications", [])
            if not row.get("batchId")
        )
    )


def verify_batch_policy(policy, spec):
    """Explicit immutable batch choices must agree with the submission receipt.

    Older frozen batches omitted this field and legitimately inherit their
    choice from the experiment receipt; do not reinterpret that omission.
    """
    if spec.get("predictorPolicy") is not None and (
        ExperimentPredictorPolicy.model_validate(spec["predictorPolicy"]).model_dump() != policy
    ):
        raise StorageError(
            "The submitted predictor choice differs from this batch's frozen predictor policy.",
            "EXPERIMENT_PREDICTOR_PLAN_CHANGED",
            409,
        )
    return policy


def policy_for_batch(submission, batch_id, *, spec=None):
    if not has_predictor_intent(submission):
        return None
    policies = submission_policies(submission)
    if batch_id not in policies:
        raise StorageError(
            "This batch is outside the submitted predictor plan.",
            "EXPERIMENT_PREDICTOR_POLICY_LOCKED",
            409,
        )
    policy = policies[batch_id]
    return (
        verify_batch_policy(policy, spec)
        if spec is not None and "predictorPolicies" in submission
        else policy
    )
