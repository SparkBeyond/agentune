"""Zero-shot adversarial tester implementation using a language model."""

import logging
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                                  SystemMessagePromptTemplate)
from langchain_core.runnables import Runnable

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

    def __init__(self, model: BaseChatModel, max_concurrency: int = 50):
        """Initializes the adversarial tester.

        Args:
            model: The language model to use for evaluation.
            max_concurrency: The maximum number of concurrent requests to the model.
        """
        self.model = model
        self.max_concurrency = max_concurrency
        self._chain = self._create_adversarial_chain()

    def _create_adversarial_chain(self) -> Runnable:
        """Creates the LangChain runnable for adversarial evaluation."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT_TEMPLATE),
        ])
        return prompt | self.model | JsonOutputParser()

    async def identify_real_conversation(
        self,
        real_conversation: Conversation,
        simulated_conversation: Conversation,
    ) -> bool:
        """Evaluate whether the language model can identify the real conversation."""
        if not real_conversation.messages or not simulated_conversation.messages:
            logger.warning("Cannot evaluate empty conversations.")
            return False

        prompt_inputs, real_label = create_comparison_prompt_inputs(
            real_conversation, simulated_conversation
        )

        try:
            parsed_response = await self._chain.ainvoke(prompt_inputs)
            identified_label = parsed_response.get("real_conversation")

            if not isinstance(identified_label, str) or identified_label not in ("A", "B"):
                logger.warning(
                    f"LLM returned an invalid value for `real_conversation`: {identified_label}"
                )
                return False

            return identified_label == real_label
        except Exception:
            logger.exception(
                "Failed to evaluate conversation pair due to an exception from the LLM chain."
            )
            return False

    async def identify_real_conversations(
        self,
        real_conversations: List[Conversation],
        simulated_conversations: List[Conversation],
    ) -> List[bool]:
        """
        Evaluate in batch whether the language model can identify the real conversations
        using an efficient batching mechanism.

        Args:
            real_conversations: A list of real conversations.
            simulated_conversations: A list of simulated conversations.

        Returns:
            A list of booleans indicating if the real conversation was identified for each pair.
        """
        if len(real_conversations) != len(simulated_conversations):
            raise ValueError(
                "Input lists for real and simulated conversations must have the same length."
            )

        if not real_conversations:
            return []

        # Prepare inputs and identify pairs that are invalid from the start
        prompt_inputs_list = []
        real_labels = []
        valid_indices = []
        for i, (real_conv, sim_conv) in enumerate(
            zip(real_conversations, simulated_conversations)
        ):
            # Only process pairs where both conversations have messages
            if real_conv.messages and sim_conv.messages:
                prompt_inputs, real_label = create_comparison_prompt_inputs(
                    real_conv, sim_conv
                )
                prompt_inputs_list.append(prompt_inputs)
                real_labels.append(real_label)
                valid_indices.append(i)
            else:
                logger.warning(
                    f"Skipping evaluation for pair {i} due to an empty conversation."
                )

        # If no pairs are valid, return a list of False
        if not valid_indices:
            return [False] * len(real_conversations)

        # Run batch evaluation on valid pairs
        try:
            llm_outputs = await self._chain.abatch(
                prompt_inputs_list, config={"max_concurrency": self.max_concurrency}
            )
        except Exception:
            logger.exception("The `abatch` call failed entirely.")
            return [False] * len(real_conversations)

        # Process results and map them back to the original list size
        final_results = [False] * len(real_conversations)
        for i, (output, real_label) in enumerate(zip(llm_outputs, real_labels)):
            original_index = valid_indices[i]

            if isinstance(output, Exception):
                logger.error(
                    f"LLM chain raised an exception for item at original index {original_index}.",
                    exc_info=output,
                )
                # Keep the result as False
                continue

            identified_label = output.get("real_conversation")
            if isinstance(identified_label, str) and identified_label in ("A", "B"):
                final_results[original_index] = identified_label == real_label
            else:
                logger.warning(
                    f"LLM returned an invalid value for `identified_as_real` for pair {original_index}: {identified_label}"
                )

        return final_results
