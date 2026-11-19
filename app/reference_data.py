from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import BusinessPolicy, Prompt


class BusinessPolicyNotFound(LookupError):
    """The singleton business policy is missing."""


class PromptNotFound(LookupError):
    """There is no prompt with the requested key."""


def get_business_policy(session: Session) -> str:
    with session.no_autoflush:
        policy = session.get(BusinessPolicy, 1)
    if policy is None:
        raise BusinessPolicyNotFound("The current business policy is missing.")
    return policy.content


def get_prompt(session: Session, prompt_key: str) -> str:
    with session.no_autoflush:
        prompt = session.get(Prompt, prompt_key)
    if prompt is None:
        raise PromptNotFound(prompt_key)
    return prompt.content


def list_prompt_keys(session: Session) -> list[str]:
    with session.no_autoflush:
        return list(session.scalars(select(Prompt.prompt_key).order_by(Prompt.prompt_key)))
