from Mapeo_Tablas import *
from Metodos import *

db = 'python Bucle_Consultas.py'

while True:
    print(tables_string)
    table = int(input("Enter the table num: "))

    if table == 1:  # Category
        print('You are operate in table Category')
        print(operations_string)
        operation = input('Insert operation: ')
        if operation == 'cr':
            run_async(create_operation(table))

    elif table == 2:  # Product
        print('You are operate in table Product')
        print(operations_string)
        operation = input('Insert operation: ')
        if operation == 'cr':
            run_async(create_operation(table))

    elif table == 3:  # Item
        print('You are operate in table Item')
        print(operations_string)
        operation = input('Insert operation: ')
        if operation == 'cr':
            run_async(create_operation(table))

    elif table == 4:  # Order
        print('You are operate in table Order')
        print(operations_string)
        operation = input('Insert operation: ')
        if operation == 'cr':
            run_async(create_operation(table))

    elif table == 5:  # Customer
        print('You are operate in table Customer')
        print(operations_string)
        operation = input('Insert operation: ')
        if operation == 'cr':
            run_async(create_operation(table))

    elif table == 6:  # Store
        print('You are operate in table Store')
        print(operations_string)
        operation = input('Insert operation: ')
        if operation == 'cr':
            run_async(create_operation(table))

    elif table == 7:  # Stock
        print('You are operate in table Stock')
        print(operations_string)
        operation = input('Insert operation: ')
        if operation == 'cr':
            run_async(create_operation(table))

    elif table == 9:
        break
