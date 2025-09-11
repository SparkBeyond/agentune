import datetime
import logging
import random

import duckdb
import pytest

from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.join.conversation import Conversation, ConversationJoinStrategy, Message

_logger = logging.getLogger(__name__)


def test_conversation_join_strategy() -> None:
    with duckdb.connect(':memory:') as conn:
        conn.execute('create table main(id integer)')
        conn.execute('create table conversation(conv_id integer, timestamp timestamp, role varchar, content varchar)')

        def insert_conversation(id: int, conversation: Conversation) -> None:
            conn.execute('insert into main(id) values ($1)', [id])
            conn.executemany('insert into conversation(conv_id, timestamp, role, content) values ($1, $2, $3, $4)',
                [[id, m.timestamp, m.role, m.content] for m in conversation.messages]
            )

        rnd = random.Random(42)

        def random_conversation() -> tuple[int, Conversation]:
            id = rnd.randint(1, 100000000)
            message_count = rnd.randint(1, 10)
            messages = tuple(
                Message(
                    rnd.choice(['user', 'assistant']),
                    datetime.datetime.fromtimestamp(rnd.randint(0, 10000000)),
                    str(rnd.random())
                )
                for _ in range(message_count)
            )
            sorted_messages = tuple(sorted(messages, key=lambda m: m.timestamp))
            return id, Conversation(sorted_messages)

        conversations = dict(random_conversation() for _ in range(100))
        for id, conversation in conversations.items():
            insert_conversation(id, conversation)

        assert len(set(conversations.values())) == len(conversations), 'Sanity check: created different conversations'
        
        main_table = DuckdbTable.from_duckdb('main', conn)
        context_table = DuckdbTable.from_duckdb('conversation', conn)

        strategy = ConversationJoinStrategy[int](
            'conversations',
            context_table,
            main_table.schema['id'],
            context_table.schema['conv_id'],
            context_table.schema['timestamp'],
            context_table.schema['role'],
            context_table.schema['content'],
        )
        strategy.index.create(conn, strategy.table.name)

        for id, conversation in conversations.items():
            assert strategy.get_conversation(conn, id) == conversation
        
        assert 1000 not in conversations
        assert strategy.get_conversation(conn, 1000) is None

        conversation_ids = list(conversations.keys())
        shuffled_ids = conversation_ids.copy()
        rnd.shuffle(shuffled_ids)
        for ids in [ [], [1000], [ rnd.choice(conversation_ids), 1000 ], rnd.choices(conversation_ids, k=20), shuffled_ids ]:
            convs = strategy.get_conversations(ids, conn)
            expected = tuple(conversations.get(id) for id in ids)
            assert convs == expected

        # Test with the role being an enum type

        strategy.index.drop(conn)
        conn.execute("alter table conversation alter role type enum('user', 'assistant')")
        context_table2 = DuckdbTable.from_duckdb('conversation', conn)
        assert context_table2.schema['role'] != context_table.schema['role']
        context2 = ConversationJoinStrategy[int](
            'conversations',
            context_table2,
            main_table.schema['id'],
            context_table2.schema['conv_id'],
            context_table2.schema['timestamp'],
            context_table2.schema['role'],
            context_table2.schema['content'],
        )
        for id, conversation in conversations.items():
            assert context2.get_conversation(conn, id) == conversation

        # Test with the role being some other type: should fail
        conn.execute('update conversation set role = null')
        conn.execute('alter table conversation alter role type int')
        context_table3 = DuckdbTable.from_duckdb('conversation', conn)
        with pytest.raises(ValueError, match='validator'):
            ConversationJoinStrategy[int](
                'conversations',
                context_table3,
                main_table.schema['id'],
                context_table3.schema['conv_id'],
                context_table3.schema['timestamp'],
                context_table3.schema['role'],
                context_table3.schema['content'],
            )


