import pathlib
from aiida import orm, engine
from aiida_shell import launch_shell_job

#function that returns the first number found in a file
def parser(dirpath: pathlib.Path):
    with open(dirpath / 'sum.out') as f:
        return {'result':orm.Float(f.read().split()[-1])}

@engine.calcfunction
def sum_and_multiplication(sum1, sum2, multiply):
    return orm.Float((sum1.value + sum2.value) * multiply.value)

@engine.workfunction
def myworkfunction(filepath1,filepath2,multiply):
    with open(filepath1.value,'r') as f1:
        content1 = f1.read()
    results1, node1 = launch_shell_job(
        orm.load_code('my_code@localhost'),
        arguments= "{input}",
        nodes={
            'input': orm.SinglefileData.from_string(content1, filename="example.txt")
        },
        outputs = ['sum.out'],
        parser = parser
    )
    with open(filepath2.value,'r') as f2:
        content2 = f2.read()
    results2, node2 = launch_shell_job(
        orm.load_code('my_code@localhost'),
        arguments= "{input}",
        nodes={
            'input': orm.SinglefileData.from_string(content2, filename="example.txt")
        },
        outputs = ['sum.out'],
        parser = parser
    )
    final_result= sum_and_multiplication(results1['result'],results2['result'],multiply)    
    return final_result #orm.Int(results1['sum.out'])*multiply.value + orm.Int(results2['sum.out'])
