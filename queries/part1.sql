SELECT 
	products.product_id AS productId,
	SUM(products.weight * orders_products.quantity) AS totalWeight
INTO OUTFILE 'C:\Users\konta\OneDrive\Documents\Karol_Dzierzak.csv'
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
FROM products
JOIN orders_products ON orders_products.product_id=products.product_id
JOIN orders ON orders.order_id = orders_products.order_id
JOIN route_segments ON route_segments.order_id = orders.order_id
WHERE 
	orders.customer_id = 32 AND route_segments.segment_type = 'STOP' AND DATE(route_segments.segment_end_time) = '2024-02-13'
GROUP BY 
	products.product_id
ORDER BY
	totalWeight;