-- 1. Producto -> Categoria
alter table producto 
add constraint fk_categoria foreign key (categoria_id) references categoria(id)


-- 2. Stock -> Sucursal & Producto
alter table stock 
add constraint fk_sucursal foreign key (sucursal_id) references sucursal(id)

alter table stock 
add constraint fk_producto foreign key (producto_id) references producto(id)


-- 3. Orden -> Cliente & Sucursal
alter table orden
add constraint fk_cliente foreign key (cliente_id) references cliente(id)

alter table orden
add constraint fk_sucursal foreign key (sucursal_id) references sucursal(id)


-- 4. Item -> Orden & Producto
alter table item
add constraint fk_orden foreign key (orden_id) references orden(id)

alter table item
add constraint fk_producto foreign key (producto_id) references producto(id)