from Modelos import *

tables_string = """
You can operate in the next tables:
* Category ---> 1
* Product  ---> 2
* Item     ---> 3
* Order    ---> 4
* Customer ---> 5
* Store    ---> 6
* Stock    ---> 7
"""
operations_string = """
The operations you can execute on the db are:
* Create    ---> cr
* To list   ---> tl
* Obtain    ---> ob
* Eliminate ---> el
"""


'''Operation Create'''
async def create_operation(table_num):
    if table_num == 1:   # Category
        name_category = input('Enter the name of the new category: ')
        new_category = await Category.create(name=name_category)
        print(f'Category create: id={new_category.id}, name={new_category.name}')

    elif table_num == 2:   # Product
        name_pro, brand_pro, unit_price_pro = list(map(str, input().split()))
        int(unit_price_pro)
        new_product = await Product.create(name=name_pro, brand=brand_pro, unit_price=unit_price_pro)
        print(f'Product create: name={new_product.name}, brand={new_product.brand},'
              f' unit_price={new_product.unit_price}')

    elif table_num == 3:  # Item
        amount_it, sale_amount_it, product_id_it, order_id_it = list(map(int, input().split()))
        new_item = await Item.create(amount=amount_it, sale_amount=sale_amount_it,
                                     product=product_id_it, order=order_id_it)
        print(f'Item create: amount={new_item.amount}, sale_amount={new_item.sale_amount},'
              f' product_id={product_id_it}, order_id={order_id_it}')

    elif table_num == 4:  # Order
        date_or, total_or, store_id_or, customer_id_or = list(map(str, input().split()))
        int(total_or), int(store_id_or), int(customer_id_or)
        new_order = await Order.create(date=date_or, total=total_or, store=store_id_or, customer=customer_id_or)
        print(f'Order create: date={new_order.date}, total={new_order.total},'
              f' store_id={store_id_or}, customer_id={customer_id_or}')

    elif table_num == 5:  # Customer
        name_cus, phone_cus = list(map(str, input().split()))
        new_customer = await Customer.create(name=name_cus, phone=phone_cus)
        print(f'Customer create: id={new_customer.id}, name={new_customer.name}, phone={new_customer.phone}')

    elif table_num == 6:  # Store
        name_s, address_s = list(map(str, input().split()))
        new_name = await Store.create(name=name_s, address=address_s)
        print(f'Store create: id={new_name.id}, name={new_name.name}, address={new_name.address}')

    elif table_num == 7:  # Stock
        amount_sk, store_id_sk, product_id_sk = list(map(int, input().split()))
        new_stock = await Stock.create(amount=amount_sk, store=store_id_sk, product=product_id_sk)
        print(f'Stock create: id={new_stock.id}, amount={new_stock.amount},'
              f' store_id={store_id_sk}, product_id={product_id_sk}')
