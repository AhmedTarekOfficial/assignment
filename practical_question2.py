import os
from openai import AsyncOpenAI
from agents import (
    Agent, Runner, function_tool,
    OpenAIChatCompletionsModel, set_tracing_disabled,
)

set_tracing_disabled(True)  # tracing would try to upload to OpenAI

client = AsyncOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

# 1. Function tool
EXAM_DATES = {
    "mathematics": "December 10, 2026",
    "physics": "December 14, 2026",
    "chemistry": "December 18, 2026",
}

@function_tool
def get_exam_date(course: str) -> str:
    """Look up the exam date of a course.

    Args:
        course: The name of the course, e.g. "Mathematics".
    """
    date = EXAM_DATES.get(course.strip().lower())
    if date is None:
        available = ", ".join(name.title() for name in EXAM_DATES)
        return f"Sorry, I couldn't find an exam date for '{course}'. Available courses: {available}."
    return f"The {course.title()} exam is on {date}."

# 2. Agent that uses the tool
exam_helper = Agent(
    name="Exam Helper",
    instructions=(
        "You help students with exam dates. Always use the get_exam_date tool "
        "to answer. If the tool says the course was not found, tell the student "
        "clearly and do not guess a date."
    ),
    model=OpenAIChatCompletionsModel(
        model="meta-llama/Llama-3.1-8B-Instruct",
        openai_client=client,
    ),
    tools=[get_exam_date],
)

# 3. Run two questions: one found, one not found
questions = [
    "When is the Physics exam?",
    "When is the Biology exam?",
]

for q in questions:
    result = Runner.run_sync(exam_helper, q)
    print(f"Q: {q}")
    print(f"A: {result.final_output}\n")
