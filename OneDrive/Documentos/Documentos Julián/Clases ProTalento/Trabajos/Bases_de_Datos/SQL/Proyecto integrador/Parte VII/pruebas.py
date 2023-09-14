from Mapeo_Tablas import *
from Modelos import *
from Metodos import *
from tortoise.functions import Count


#  run_async(obtain_operation(2))

async def prueba_str(table, num):
    register = await table.get_or_none(id=num).first()
    print(register)

run_async(prueba_str(Customer, 1))
