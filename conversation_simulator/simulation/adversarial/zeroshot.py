"""Zero-shot adversarial tester implementation using a language model."""

import logging
import random

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                                  SystemMessagePromptTemplate)
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda

from ...models.conversation import Conversation
from .base import AdversarialTester
from .prompts import (
    HUMAN_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    create_comparison_prompt_inputs,
)

logger = logging.getLogger(__name__)


class ZeroShotAdversarialTester(AdversarialTester):
    """Zero-shot adversarial tester using a language model and a structured parser."""

    def __init__(self, model: BaseChatModel, max_concurrency: int = 50, random_seed: int = 0):
        """Initializes the adversarial tester.

        Args:
            model: The language model to use for evaluation.
            max_concurrency: The maximum number of concurrent requests to the model.
            random_seed: Optional seed for reproducible random number generation.
        """
        self.model = model
        self.max_concurrency = max_concurrency
        self._random = random.Random(random_seed)
        self._chain = self._create_adversarial_chain()

    @staticmethod
    def _extract_label(output: dict) -> str | None:
        """Extracts the identified conversation label from the model output."""
        identified_label = output.get("real_conversation")
        if not isinstance(identified_label, str) or identified_label not in ("A", "B"):
            logger.warning(f"LLM returned invalid value: {identified_label}")
            return None
        return identified_label

    def _create_adversarial_chain(self) -> Runnable:
        """Creates the LangChain runnable for adversarial evaluation."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT_TEMPLATE),
        ])

        label_extractor = RunnableLambda(self._extract_label)
        return prompt | self.model | JsonOutputParser() | label_extractor


    async def identify_real_conversation(
        self, real_conversation: Conversation, simulated_conversation: Conversation
    ) -> bool | None:
        """Identify which conversation is real.

        Args:
            real_conversation: A real conversation.
            simulated_conversation: A simulated conversation.

        Returns:
            bool: True if the real conversation was identified, False if the simulated
                conversation was identified, None if either conversation is empty or invalid.
        """
        results = await self.identify_real_conversations(
            [real_conversation], [simulated_conversation]
        )
        return results[0] if results else None

    async def identify_real_conversations(
        self,
        real_conversations: list[Conversation],
        simulated_conversations: list[Conversation],
    ) -> list[bool | None]:
        """Evaluate multiple conversation pairs in batch.

        Args:
            real_conversations: List of real conversations.
            simulated_conversations: List of simulated conversations.

        Returns:
            List of results, where each element is True if the real conversation was
                identified, False if the simulated conversation was identified, or None
                if either conversation was empty or an error occurred.
        """
        if len(real_conversations) != len(simulated_conversations):
            raise ValueError(
                "real_conversations and simulated_conversations must have the same length"
            )

        if not real_conversations:
            return []

        # Prepare inputs and track which indices are valid
        prompt_inputs = []
        valid_indices = []
        real_labels = []
        results: list[bool | None] = [None] * len(real_conversations)

        for i, (real_conv, sim_conv) in enumerate(zip(real_conversations, simulated_conversations)):
            if not real_conv.messages or not sim_conv.messages:
                continue  # Skip empty conversations, leaving as None

            is_real_a = self._random.choice([True, False])
            prompt_inputs.append(
                create_comparison_prompt_inputs(real_conv, sim_conv, is_real_a)
            )
            real_labels.append("A" if is_real_a else "B")
            valid_indices.append(i)

        if not prompt_inputs:
            return results

        # Process valid conversations in batch
        try:
            identified_labels = await self._chain.abatch(prompt_inputs, max_concurrency=self.max_concurrency)
            
            for idx, identified_label, real_label in zip(valid_indices, identified_labels, real_labels):
                if identified_label is None:
                    continue
                results[idx] = identified_label == real_label

        except Exception as e:
            logger.error("Batch processing failed", exc_info=e)
            # Leave results as None for failed items

        return results
