import datetime
import random

import duckdb

from agentune.analyze.context.conversation import Conversation, ConversationContext, Message
from agentune.analyze.core.database import DuckdbTable


def test_conversation_context() -> None:
    with duckdb.connect(':memory:lookup') as conn:
        conn.execute('create table main(id integer)')
        conn.execute('create table conversation(id integer, timestamp timestamp, role varchar, content varchar)')

        def insert_conversation(id: int, conversation: Conversation) -> None:
            conn.execute('insert into main(id) values ($1)', [id])
            conn.executemany('insert into conversation(id, timestamp, role, content) values ($1, $2, $3, $4)', 
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

        context = ConversationContext[int](
            'conversations',
            context_table,
            main_table.schema['id'],
            context_table.schema['id'],
            context_table.schema['timestamp'],
            context_table.schema['role'],
            context_table.schema['content'],
        )
        context.index.create(conn, context.table.name)

        for id, conversation in conversations.items():
            assert context.get_conversation(conn, id) == conversation
        
        assert 1000 not in conversations
        assert context.get_conversation(conn, 1000) is None

        for ids in [ [], [1], [1000], [1, 5, 58, 18, 43, 101, 30, 502, 8] ]:
            assert context.get_conversations(ids, conn) == tuple(conversations.get(id) for id in ids)
        
