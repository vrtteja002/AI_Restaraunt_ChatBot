from langchain.memory import ConversationBufferMemory

MEMORY = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key='answer'
)