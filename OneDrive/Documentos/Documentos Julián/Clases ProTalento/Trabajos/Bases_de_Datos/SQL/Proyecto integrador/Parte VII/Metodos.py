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

'''CREATE METHODS'''
async def create_category(name_category: str):
    category_ins = await Category.create(name=name_category)
    print(f'Category create: id={category_ins.id}, name={category_ins.name}')

async def create_product(name_p: str, brand_p: str, unit_price_p: int, category_fk: int):
    product_ins = await Product.create(name=name_p, brand=brand_p, unit_price=unit_price_p, category=category_fk)
    print(f'Product create: name={product_ins.name}, brand={product_ins.brand}, unit_price={product_ins.unit_price},'
          f' category_id={category_fk} ')

async def create_item(amount_i: int, sale_amount_i: int, product_fk: int, order_fk: int):
    item_ins = await Item.create(amount=amount_i, sale_amount=sale_amount_i, product=product_fk, order=order_fk)
    print(f'Item create: amount={item_ins.amount}, sale_amount={item_ins.sale_amount}, product_id={product_fk},'
          f'order_id={order_fk}')

async def create_order(date_o: str, total_o: int, store_fk: int, customer_fk: int):
    order_ins = await Order.create(date=date_o, total=total_o, store=store_fk, customer=customer_fk)
    print(f'Order create: date={order_ins.date}, total={order_ins.total}, store_id={store_fk},'
          f' customer_id={customer_fk}')

async def create_customer(name_c: str, phone_c: str):
    customer_ins = await Customer.create(name=name_c, phone=phone_c)
    print(f'Customer create: id={customer_ins.id}, name={customer_ins.name}, phone={customer_ins.phone}')

async def create_store(name_s: str, address_s: str):
    store_ins = await Store.create(name=name_s, address=address_s)
    print(f'Store create: id={store_ins.id}, name={store_ins.name}, address={store_ins.address}')

async def create_stock(amount_stock: int, store_fk: int, product_fk: int):
    stock_ins = await Stock.create(amount=amount_stock, store=store_fk, product=product_fk)
    print(f'Stock create: id={stock_ins.id}, amount={stock_ins.amount}, store_id={store_fk}, product_id={product_fk}')
