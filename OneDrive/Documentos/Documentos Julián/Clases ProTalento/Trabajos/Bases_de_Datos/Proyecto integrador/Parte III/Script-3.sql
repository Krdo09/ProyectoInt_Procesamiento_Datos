create table producto(
	id serial primary key,
	nombre varchar(50),
	marca varchar(50),
	precio_unitario integer,
	categoria_id integer
);

create table categoria(
	id serial primary key,
	nombre varchar(50)
);

create table item(
	id serial primary key,
	cantidad integer,
	monto_venta integer,
	orden_id integer,
	producto_id integer
);

create table orden(
	id serial primary key,
	fecha timestamp,
	total integer,
	cliente_id integer,
	sucursal_id integer
);

create table cliente(
	id serial primary key,
	nombre integer,
	telefono varchar(50)
);

create table sucursal(
	id serial primary key,
	nombre varchar(50),
	direccion varchar(100)
);

create table stock(
	id serial primary key,
	cantidad integer,
	sucursal_id integer not null,
	producto_id integer not null,
	unique (sucursal_id, producto_id)
);