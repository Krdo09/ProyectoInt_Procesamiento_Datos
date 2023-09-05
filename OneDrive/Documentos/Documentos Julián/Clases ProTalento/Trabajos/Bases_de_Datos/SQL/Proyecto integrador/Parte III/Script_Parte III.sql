create table producto(
	id serial primary key,
	nombre varchar(50),
	marca varchar(50),
	categoria_id integer,
	precio_unitario integer
);

create table categoria(
	id serial primary key,
	nombre varchar(50)
);

create table item(
	id serial primary key,
	orden_id integer,
	producto_id integer,
	cantidad integer,
	monto_venta integer
);

create table orden(
	id serial primary key,
	cliente_id integer,
	sucursal_id integer,
	fecha timestamp,
	total integer
);

create table cliente(
	id serial primary key,
	nombre varchar(50),
	telefono varchar(50)
);

create table sucursal(
	id serial primary key,
	nombre varchar(50),
	direccion varchar(100)
);

create table stock(
	id serial primary key,
	sucursal_id integer not null,
	producto_id integer not null,
	cantidad integer,
	unique (sucursal_id, producto_id)
);

 --drop table item 