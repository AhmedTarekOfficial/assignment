from agents import Agent, Runner, ModelSettings


study_buddy = Agent(
    name="Study Buddy",
    instructions=(
        "Explain any topic the user gives you in exactly 3 short bullet points. "
        "Do not add any intro, outro, or extra text."
    ),
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=0.0, max_tokens=100),
)

# 3. Read a topic, run the agent synchronously, and print the result
topic = input("Enter a topic: ")
result = Runner.run_sync(study_buddy, topic)
print(result.final_output)

