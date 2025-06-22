from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class LlmUtils:
    @staticmethod
    def batch_llm_generate(llm, prompts, user_prefix="statement: "):
        """
        prompts: list of (prompt, statement) pairs
        Returns: list of responses (strings)
        """
        messages_batch = []
        for prompt, statement in prompts:
            full_prompt = f"{prompt}\n\n{user_prefix}{statement}"
            messages_batch.append([HumanMessage(content=full_prompt)])
        responses = llm.generate(messages_batch)
        return [gen.generations[0][0].text.strip() for gen in responses.generations]
