from aiida import orm
from aiida_shell import launch_shell_job


content = """we sum 10.12 and 12 to see
if we get something that is not 2
"""

results, node = launch_shell_job(
    orm.load_code('my_code@localhost'),
    arguments= "{input}",
    nodes={
        'input': orm.SinglefileData.from_string(content, filename="example.txt")
    },
    outputs = ['sum.out']
)
