from agentune.analyze.context.base import TablesWithContextDefinitions, TableWithContextDefinitions
from agentune.analyze.context.conversation import ConversationContext
from agentune.analyze.core.database import DuckdbManager, DuckdbTable


def test_context_table_helpers(ddb_manager: DuckdbManager) -> None:
    with ddb_manager.cursor() as conn:
        conn.execute('create table main(id integer)')
        conn.execute('create table conversation(conv_id integer, timestamp timestamp, role varchar, content varchar)')
        conn.execute('create table conversation2(conv_id integer, timestamp timestamp, role varchar, content varchar)')

        main_table = DuckdbTable.from_duckdb('main', conn)
        context_table = DuckdbTable.from_duckdb('conversation', conn)
        context_table2 = DuckdbTable.from_duckdb('conversation2', conn)

        context = ConversationContext[int](
            'conversations',
            context_table,
            main_table.schema['id'],
            context_table.schema['conv_id'],
            context_table.schema['timestamp'],
            context_table.schema['role'],
            context_table.schema['content'],
        )
        context2 = ConversationContext[int](
            'conversations2',
            context_table,
            main_table.schema['id'],
            context_table.schema['conv_id'],
            context_table.schema['timestamp'],
            context_table.schema['role'],
            context_table.schema['content'],
        )
        context3 = ConversationContext[int](
            'conversations3',
            context_table2,
            main_table.schema['id'],
            context_table.schema['conv_id'],
            context_table.schema['timestamp'],
            context_table.schema['role'],
            context_table.schema['content'],
        )

        contexts = TablesWithContextDefinitions.group([context, context2, context3])
        assert contexts == TablesWithContextDefinitions({
            context_table.name: TableWithContextDefinitions(context_table, {
                context.name: context,
                context2.name: context2
            }),
            context_table2.name: TableWithContextDefinitions(context_table2, {
                context3.name: context3
            }),
        })

