from tortoise.models import Model
from tortoise import fields

class Category(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)


class Product(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)
    brand = fields.CharField(max_length=255)
    unit_price = fields.IntField()

    category = fields.ForeignKeyField('models.Category', related_name='categories')


class Item(Model):
    id = fields.IntField(pk=True)
    amount = fields.IntField()
    sale_amount = fields.IntField

    product = fields.ForeignKeyField('models.Product', related_name='products')
    order = fields.ForeignKeyField('models.Order', related_name='orders')


class Order(Model):
    id = fields.IntField(pk=True)
    date = fields.DatetimeField()
    total = fields.IntField()

    store = fields.ForeignKeyField('models.Store', related_name='stores')
    customer = fields.ForeignKeyField('models.Customer', related_name='customers')

    def __str__(self):
        return f'id={self.id}, date={self.date}, total={self.total}, store_id={self.store}, customer_id={self.customer}'

    def __repr__(self):
        return self.__str__()


class Customer(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)
    phone = fields.CharField(max_length=55)


class Store(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=55)
    address = fields.CharField(max_length=255)


class Stock(Model):
    id = fields.IntField(pk=True)
    amount = fields.IntField()

    store = fields.ForeignKeyField('models.Store', related_name='stores_stock')
    product = fields.ForeignKeyField('models.Product', related_name='products_stock')
