from tortoise.models import Model
from tortoise import fields

class Category(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)

    def __str__(self):
        return f'Category: id={self.id}, name={self.name}'


class Product(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)
    brand = fields.CharField(max_length=255)
    unit_price = fields.IntField()
    category = fields.ForeignKeyField('models.Category', related_name='categories')
    
    def __str__(self):
        return (f'Product found: id={self.id}, name={self.name}, brand={self.brand},'
                f' unit price={self.unit_price}, category={self.category}')


class Item(Model):
    id = fields.IntField(pk=True)
    amount = fields.IntField()
    sale_amount = fields.IntField
    product = fields.ForeignKeyField('models.Product', related_name='products')
    order = fields.ForeignKeyField('models.Order', related_name='orders')

    def __str__(self):
        return (f'Item: id={self.id}, amount={self.amount}, sale_amount={self.sale_amount},'
                f' product={self.product}, order={self.order}')


class Order(Model):
    id = fields.IntField(pk=True)
    date = fields.DatetimeField()
    total = fields.IntField()
    store = fields.ForeignKeyField('models.Store', related_name='stores')
    customer = fields.ForeignKeyField('models.Customer', related_name='customers')

    def __str__(self):
        return (f'Order: id={self.id}, date={self.date}, total={self.total},'
                f' store={self.store}, customer={self.customer}')


class Customer(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)
    phone = fields.CharField(max_length=55)

    def __str__(self):
        return f'Customer: id={self.id}, name={self.name}, phone={self.phone}'


class Store(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=55)
    address = fields.CharField(max_length=255)

    def __str__(self):
        return f'Store: id={self.id}, name={self.name}, address={self.address}'


class Stock(Model):
    id = fields.IntField(pk=True)
    amount = fields.IntField()
    store = fields.ForeignKeyField('models.Store', related_name='stores_stock')
    product = fields.ForeignKeyField('models.Product', related_name='products_stock')

    def __str__(self):
        return f'Stock: id={self.id}, amount={self.amount}, store={self.store}, product={self.product}'
