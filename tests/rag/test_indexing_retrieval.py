"""Tests for indexing and retrieval functionality.

This test suite provides black box tests for the indexing and retrieval modules
using InMemoryVectorStore and DeterministicFakeEmbedding.
"""

import abc
import pytest
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_community.vectorstores import FAISS

from conversation_simulator.models import Conversation, Message, ParticipantRole
from conversation_simulator.rag.indexing_and_retrieval import (
    conversations_to_langchain_documents,
    get_similar_finished_conversations,
    get_similar_examples_for_next_message_role
)
from tests.rag.test_data import create_test_conversations


class BaseIndexingRetrievalTests(abc.ABC):
    """Base test class for indexing and retrieval tests.
    
    This class defines a common set of test scenarios for vector store implementations.
    Subclasses should override init_vector_store() to test different implementations.
    """
    
    def setup_method(self):
        """Set up test data for each test method."""
        # Initialize embedding function
        self.embedding_func = DeterministicFakeEmbedding(size=10)

        # Get test conversations
        self.test_conversations = create_test_conversations()
        
        # Convert conversations to documents
        self.test_documents = conversations_to_langchain_documents(self.test_conversations)
        
        # Initialize a vector store
        self.vector_store = self.init_vector_store(self.embedding_func, self.test_documents)
    
    @abc.abstractmethod
    def init_vector_store(self, embedding_function, test_documents: list[Document]) -> VectorStore:
        """Abstract factory method to create an empty vector store instance.
        
        Subclasses must implement this to test different vector store implementations.
        
        Args:
            embedding_function: The embedding function to use for the vector store
            test_documents: List of documents to initialize the vector store with
            
        Returns:
            An empty vector store instance
        """
        pass
    
    def assert_docs_match_criteria(self, docs, expected_count, filter_criteria=None, message=None):
        """
        Validate that the retrieved documents match expected criteria.
        
        Args:
            docs: List of documents retrieved from the vector store
            expected_count: Number of documents expected (usually k parameter)
            filter_criteria: Dict of metadata fields and values that all docs should have
            message: Optional message prefix for assertions
        """
        message = message or "Document validation failed"
        
        # Check document count
        assert len(docs) == expected_count, f"{message}: Expected {expected_count} docs, got {len(docs)}"
        
        # If filter criteria provided, check that all documents match it
        if filter_criteria:
            for i, doc in enumerate(docs):
                for key, value in filter_criteria.items():
                    assert doc.metadata.get(key) == value, f"{message}: Doc at index {i} failed filter check for {key}={value}"
    
    def test_conversation_to_documents_conversion(self):
        """Test conversion of conversations to documents."""
        # Test that empty conversations are filtered
        empty_conv = Conversation(messages=(), outcome=None)
        docs = conversations_to_langchain_documents([empty_conv])
        assert len(docs) == 0, "Empty conversation should not generate documents"
        
        # Test simple conversation conversion
        single_message_conv = Conversation(
            messages=(
                Message(sender=ParticipantRole.CUSTOMER, content="Test message", timestamp=datetime.now()),
            ),
            outcome=None
        )
        
        docs = conversations_to_langchain_documents([single_message_conv])
        assert len(docs) == 1, "Single message conversation should generate one document"
        assert docs[0].page_content == "Customer: Test message", "Page content format is incorrect"
        assert docs[0].metadata["current_message_role"] == ParticipantRole.CUSTOMER.value
        assert docs[0].metadata["has_next_message"] is False
        
        # Test multi-message conversation conversion
        two_message_conv = Conversation(
            messages=(
                Message(sender=ParticipantRole.CUSTOMER, content="Hello", timestamp=datetime.now()),
                Message(sender=ParticipantRole.AGENT, content="Hi there", timestamp=datetime.now()),
            ),
            outcome=None
        )
        
        docs = conversations_to_langchain_documents([two_message_conv])
        assert len(docs) == 2, "Two message conversation should generate two documents"
        
        # First document represents history up to the first message
        assert docs[0].page_content == "Customer: Hello", "First document page content is incorrect"
        assert docs[0].metadata["has_next_message"] is True
        assert docs[0].metadata["next_message_role"] == ParticipantRole.AGENT.value
        assert docs[0].metadata["next_message_content"] == "Hi there"
        
        # Second document represents history up to the second message
        assert docs[1].page_content == "Customer: Hello\nAgent: Hi there", "Second document page content is incorrect"
        assert docs[1].metadata["has_next_message"] is False
    
    @pytest.mark.asyncio
    async def test_filtering_by_next_message_role(self):
        """Test filtering documents by next_message_role."""
        # Create a test history for retrieving agent examples
        history = (Message(sender=ParticipantRole.CUSTOMER, content="What are your business hours?", timestamp=datetime.now()),)
        
        # Request agent responses
        k = 2
        docs = await get_similar_examples_for_next_message_role(
            conversation_history=history,
            vector_store=self.vector_store,
            k=k,
            target_role=ParticipantRole.AGENT
        )
        
        # Check that we got expected number of results and they all have agent as next role
        self.assert_docs_match_criteria(
            docs,
            expected_count=k,
            filter_criteria={"next_message_role": ParticipantRole.AGENT.value},
            message="Agent role filtering test failed"
        )
        
        # Also test with customer role
        history = (Message(sender=ParticipantRole.AGENT, content="Is there anything else I can help you with?", timestamp=datetime.now()),)
        
        k = 2
        docs = await get_similar_examples_for_next_message_role(
            conversation_history=history,
            vector_store=self.vector_store,
            k=k,
            target_role=ParticipantRole.CUSTOMER
        )
        
        # Check that we got expected number of results and they all have customer as next role
        self.assert_docs_match_criteria(
            docs,
            expected_count=k,
            filter_criteria={"next_message_role": ParticipantRole.CUSTOMER.value},
            message="Customer role filtering test failed"
        )
    
    @pytest.mark.asyncio
    async def test_filtering_finished_conversations(self):
        """Test filtering for finished conversations."""
        # Create a partial conversation to use as query
        query_conversation = Conversation(
            messages=(
                Message(sender=ParticipantRole.CUSTOMER, content="I need help with my account", timestamp=datetime.now()),
            ),
            outcome=None
        )
        
        # Retrieve similar finished conversations
        k = 2
        docs_with_scores = await get_similar_finished_conversations(
            vector_store=self.vector_store,
            conversation=query_conversation,
            k=k
        )
        
        # Check that we got expected number of results
        assert len(docs_with_scores) <= k, f"Expected at most {k} results, got {len(docs_with_scores)}"
        
        # Verify all returned documents have has_next_message=False
        for doc, _ in docs_with_scores:
            assert doc.metadata.get("has_next_message") is False, "Document from get_similar_finished_conversations has has_next_message=True"


class TestInMemoryVectorStore(BaseIndexingRetrievalTests):
    """Test suite for InMemoryVectorStore implementation."""
    
    def init_vector_store(self, embedding_function, test_documents: list[Document]) -> VectorStore:
        """Create an empty InMemoryVectorStore instance."""
        vector_store = InMemoryVectorStore(embedding=embedding_function)
        vector_store.add_documents(test_documents)
        return vector_store


class TestFaissVectorStore(BaseIndexingRetrievalTests):
    """Test suite for FaissVectorStore implementation."""

    def init_vector_store(self, embedding_function, test_documents: list[Document]) -> VectorStore:
        """Create an empty FaissVectorStore instance."""
        return FAISS.from_documents(
            documents=test_documents,
            embedding=embedding_function
        )