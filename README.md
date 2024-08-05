# Decentralized AI Agent Task Delegation System

## Introduction

The Agent Task Delegation System represents a paradigm shift in artificial intelligence problem-solving. This innovative framework addresses the limitations of single-agent AI systems by implementing a multi-agent ecosystem powered by decentralized infrastructure.

We at [Capx AI](https://huggingface.co/Capx) are researching on systems that efficiently offloads actionable tasks to specialized agentic services within the network, enabling the handling of complex, multi-faceted problems with unprecedented efficiency and scalability.

## System Architecture
![architecture](https://cdn-uploads.huggingface.co/production/uploads/644bf6ef778ecbfb977e8e84/M5O0zNIAhON2UjDFS0f1k.png)

## Single Agent system drawbacks
Traditional single-agent AI systems face several limitations when dealing with complex tasks. The Agent Task Delegation System is designed to overcome these challenges:

- **Bandwidth Limitations**: Single agents struggle to process and communicate information across multiple tasks simultaneously.
- **Processing Power Constraints**: Individual agents face computational limits when handling diverse tasks.
- **Memory Constraints**: Difficulty in juggling different contexts for various tasks within a single agent.
- **Task Complexity**: Single agents often struggle with multi-step tasks requiring diverse skills.


## Components of a Task Delegation System
### Control Plane

The Control Plane acts as the central orchestrator, managing task allocation and system state.

```python
from llama_agents import (
    AgentService,
    AgentOrchestrator,
    ControlPlaneServer,
    SimpleMessageQueue,
)

control_plane = ControlPlaneServer(
    message_queue=message_queue,
    orchestrator=AgentOrchestrator(llm=OpenAI(model="gpt-4-turbo")),
    port=8001,
)
```

### Message Queue

The Message Queue facilitates communication between services and the Control Plane.

```python
message_queue = SimpleMessageQueue(port=8000)
```

### Orchestrator

The Orchestrator module manages task allocation and result processing.

```python
class AgentOrchestrator(BaseOrchestrator):
    def get_next_messages(self, task_def, tools, state):
        chat_history = state.get(HISTORY_KEY, [])
        memory = ChatMemoryBuffer(chat_history, self.llm)

        if not chat_history:
            response = self.llm.predict_and_call(tools, task_def.input)
        else:
            response = self.llm.predict_and_call(tools + [self.finalize_tool], memory.get())

        return queue_messages, new_state

    def add_result_to_state(self, result, state):
        chat_history = state.get(HISTORY_KEY, [])
        return updated_state
```

### Specialized Agent Services

These are the core operational units where task processing occurs.

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

def get_the_secret_fact() -> str:
    return "The secret fact is: A baby llama is called a 'Cria'."

tool = FunctionTool.from_defaults(fn=get_the_secret_fact)

agent1 = ReActAgent.from_tools([tool], llm=OpenAI())
agent2 = ReActAgent.from_tools([], llm=OpenAI())

agent_server_1 = AgentService(
    agent=agent1,
    message_queue=message_queue,
    description="Useful for getting the secret fact.",
    service_name="secret_fact_agent",
    port=8002,
)
agent_server_2 = AgentService(
    agent=agent2,
    message_queue=message_queue,
    description="Useful for getting random dumb facts.",
    service_name="dumb_fact_agent",
    port=8003,
)
```

## Implementation

### Local / Notebook Flow

For faster iteration in a notebook environment:

```python
from llama_agents import LocalLauncher
import nest_asyncio

nest_asyncio.apply()

launcher = LocalLauncher(
    [agent_server_1, agent_server_2],
    control_plane,
    message_queue,
)
result = launcher.launch_single("What is the secret fact?")

print(f"Result: {result}")
```

### Server Flow

For a more scalable, production-ready setup:

```python
from llama_agents import ServerLauncher, CallableMessageConsumer

def handle_result(message) -> None:
    print(f"Got result:", message.data)

human_consumer = CallableMessageConsumer(
    handler=handle_result, message_type="human"
)

launcher = ServerLauncher(
    [agent_server_1, agent_server_2],
    control_plane,
    message_queue,
    additional_consumers=[human_consumer],
)

launcher.launch_servers()
```

To interact with the server:

```python
from llama_agents import LlamaAgentsClient

client = LlamaAgentsClient("http://127.0.0.1:8001")
task_id = client.create_task("What is the secret fact?")
result = client.get_task_result(task_id)
```

## Benefits

1. **Improved Efficiency**: Parallel processing of subtasks.
2. **Enhanced Scalability**: Easy addition of new specialized agents and services.
3. **Increased Flexibility**: Dynamic adaptation to task requirements.
4. **Robust Problem-Solving**: Effective handling of complex, interdependent problems.

## Conclusion

The Agent Task Delegation System represents a significant advancement in AI problem-solving capabilities. By decomposing tasks and leveraging specialized agents, this method offers improved efficiency, adaptability, and scalability compared to traditional AI approaches.

---

*Acknowledgments: This project draws inspiration from various multi-agent systems and distributed computing paradigms, including the llama-agents by [Llama Index](https://www.llamaindex.ai/) framework. We extend our gratitude to the broader AI research community for their continuous innovations in this field.*
