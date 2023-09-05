from Mapeo_Tablas import *
from Metodos import *

db = 'python Bucle_Consultas.py'

while True:
    print(tables_string)
    table = int(input("Enter the table num: "))

    if table == 1: #Category
        print('You are operate in table Category')
        print(operations_string)
        operation = input('Insert operation: ')
        if operation == 'cr':
            name_category = input('Enter the name of the new category: ')
            run_async(create_category(name_category))

    elif table == 2: #Product
        print('You are operate in table Product')
        print(operations_string)
        operation = input('Insert operation: ')
        if operation == 'cr':
            name, brand, unit_price, category_id = list(map(str, input().split()))
            int(unit_price), int(category_id)
            run_async(create_product(name, brand, unit_price, category_id))

    elif table == 3: #Item
        print('You are operate in table Item')
        print(operations_string)
        operation = input('Insert operation: ')
        if operation == 'cr':
            amount, sale_amount, product_id, order_id = list(map(int, input().split()))
            run_async(create_item(amount, sale_amount, product_id, order_id))

    elif table == 4: #Order
        print('You are operate in table Order')
        print(operations_string)
        operation = input('Insert operation: ')
        if operation == 'cr':
            date, total, store_id, customer_id = list(map(str, input().split()))
            run_async(create_order(date, int(total), int(store_id), int(customer_id)))

    elif table == 5: #Customer
        print('You are operate in table Customer')
        print(operations_string)
        operation = input('Insert operation: ')
        if operation == 'cr':
            name, phone = list(map(str, input().split()))
            run_async(create_customer(name, phone))

    elif table == 6: #Store
        print('You are operate in table Store')
        print(operations_string)
        operation = input('Insert operation: ')
        if operation == 'cr':
            name, address = list(map(str, input().split()))
            run_async(create_store(name, address))

    elif table == 7: #Stock
        print('You are operate in table Stock')
        print(operations_string)
        operation = input('Insert operation: ')
        if operation == 'cr':
            amount, store_id, product_id = list(map(int, input().split()))
            run_async(create_stock(amount, store_id, product_id))

    elif table == 9:
        break
