from agentune.analyze.core.database import DuckdbManager, DuckdbTable
from agentune.analyze.join.base import TablesWithJoinStrategies, TableWithJoinStrategies
from agentune.analyze.join.conversation import ConversationJoinStrategy


def test_join_strategy_helpers(ddb_manager: DuckdbManager) -> None:
    with ddb_manager.cursor() as conn:
        conn.execute('create table main(id integer)')
        conn.execute('create table conversation(conv_id integer, timestamp timestamp, role varchar, content varchar)')
        conn.execute('create table conversation2(conv_id integer, timestamp timestamp, role varchar, content varchar)')

        main_table = DuckdbTable.from_duckdb('main', conn)
        context_table = DuckdbTable.from_duckdb('conversation', conn)
        context_table2 = DuckdbTable.from_duckdb('conversation2', conn)

        strategy = ConversationJoinStrategy[int](
            'conversations',
            context_table,
            main_table.schema['id'],
            context_table.schema['conv_id'],
            context_table.schema['timestamp'],
            context_table.schema['role'],
            context_table.schema['content'],
        )
        strategy2 = ConversationJoinStrategy[int](
            'conversations2',
            context_table,
            main_table.schema['id'],
            context_table.schema['conv_id'],
            context_table.schema['timestamp'],
            context_table.schema['role'],
            context_table.schema['content'],
        )
        context3 = ConversationJoinStrategy[int](
            'conversations3',
            context_table2,
            main_table.schema['id'],
            context_table.schema['conv_id'],
            context_table.schema['timestamp'],
            context_table.schema['role'],
            context_table.schema['content'],
        )

        join_strategies = TablesWithJoinStrategies.group([strategy, strategy2, context3])
        assert join_strategies == TablesWithJoinStrategies({
            context_table.name: TableWithJoinStrategies(context_table, {
                strategy.name: strategy,
                strategy2.name: strategy2
            }),
            context_table2.name: TableWithJoinStrategies(context_table2, {
                context3.name: context3
            }),
        })

