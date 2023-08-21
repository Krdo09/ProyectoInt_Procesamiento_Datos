-- 1. Calcular el precio promedio de los productos en cada categoría
select c.nombre as nombre_categoria, avg(p.precio_unitario) as promedio_productos
from categoria c 
join producto p on c.id = p.categoria_id 
group by c.nombre 
order by  promedio_productos desc 


-- 2. Obtener la cantidad total de productos en stock por categoría
select c.nombre as categoria_nombre, sum(s.cantidad) as total_productos
from producto p
join stock s on p.id = s.producto_id 
join categoria c on p.categoria_id = c.id
group by c.nombre
order by total_productos desc 


-- 3. Calcular el total de ventas por sucursal
select s.nombre as nombre_sucursal, sum(o.total) as total_ventas
from orden o 
join sucursal s on o.sucursal_id = s.id 
group by s.nombre 
order by total_ventas desc 


--- 4. Obtener el cliente que ha realizado el mayor monto de compras
select c.nombre, sum(o.total) as total_compras
from cliente c 
join orden o on c.id = o.cliente_id 
group by c.nombre 
limit 1