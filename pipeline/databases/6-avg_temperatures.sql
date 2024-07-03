-- Import in hbtn_0c_0 database a table dump
-- Write a script that displays the average temperature (Fahrenheit)
-- by city ordered by temperature (descending).

SELECT city, AVG(temperature) AS average_temperature
FROM temperatures
GROUP BY city
ORDER BY average_temperature DESC;
