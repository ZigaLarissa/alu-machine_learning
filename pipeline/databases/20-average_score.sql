-- Write a SQL script that creates a stored procedure
-- ComputeAverageScoreForUser that computes and store
-- the average score for a student.

-- Requirements:

-- Procedure ComputeAverageScoreForUser is taking 1 input:
-- user_id, a users.id value (you can assume user_id is linked
-- to an existing users)


DELIMITER //

CREATE PROCEDURE ComputeAverageScoreForUser (
    IN user_id INT
)

BEGIN
    DECLARE average_score DECIMAL(5, 2);

    -- Compute the average score
    SELECT AVG(score)
    INTO average_score
    FROM corrections
    WHERE user_id = user_id;

    -- Update the user
    UPDATE users
    SET average_score = average_score
    WHERE id = user_id;

END //

DELIMITER ;
