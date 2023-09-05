
--Obtener el precio mínimo, precio máximo y precio promedio de todos los productos.

select min(p.precio_unitario) as precio_minimo 
from producto p

select max(p.precio_unitario) as precio_maximo
from producto p

select avg(p.precio_unitario) as precio_promedio 
from producto p 


--Calcular la cantidad total de productos en stock por sucursal.
select s.sucursal_id, sum(cantidad) as total_productos
from stock s 
group by s.sucursal_id
order by s.sucursal_id asc


--Obtener el total de ventas por cliente.
select o.cliente_id, sum(o.total) as total_ventas_cliente
from orden o 
group by o.cliente_id
order by o.cliente_id asc 