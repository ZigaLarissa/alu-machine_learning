-- Import the database hbtn_0d_tvshows_rate dump to your MySQL server
-- Write a script that lists all shows from hbtn_0d_tvshows_rate
-- by their rating.
-- Each record should display: tv_shows.title - rating sum
-- Results must be sorted in descending order by the rating
-- You can use only one SELECT statement
-- The database name will be passed as an argument of the mysql command

SELECT tv_shows.title, tv_show_rating.rate
FROM tv_shows
JOIN tv_show_rating ON tv_shows.id = tv_show_rating.show_id
ORDER BY tv_show_rating.rate DESC;
