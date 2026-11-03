import os
from typing import List
from typing_extensions import TypedDict  # Pydantic requires this on Python < 3.12
from pydantic import BaseModel, ConfigDict
from openai import AsyncOpenAI
from agents import (
    Agent, Runner, trace,
    OpenAIChatCompletionsModel, set_tracing_disabled,
)

set_tracing_disabled(True)  # no OpenAI key, so skip uploading traces

client = AsyncOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

# 1. TypedDict with day and topic
class StudyDay(TypedDict):
    day: str
    topic: str

# 2. Pydantic model containing a list of StudyDay
class StudyPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    days: List[StudyDay]

# 3. Agent with StudyPlan as its output_type
study_planner = Agent(
    name="Study Planner",
    instructions=(
        "Create a 3-day study plan for the subject the user gives you. "
        "Return exactly 3 days, labeled 'Day 1', 'Day 2', and 'Day 3', "
        "each with a short topic."
    ),
    model=OpenAIChatCompletionsModel(
        model="Qwen/Qwen2.5-7B-Instruct",
        openai_client=client,
    ),
    output_type=StudyPlan,
)

# 4. Run inside a trace and print each day and topic
with trace("Lab 3 - Study Planner"):
    result = Runner.run_sync(study_planner, "OpenAI Agents SDK")

plan = result.final_output  # a StudyPlan instance
for item in plan.days:
    print(f"{item['day']}: {item['topic']}")
