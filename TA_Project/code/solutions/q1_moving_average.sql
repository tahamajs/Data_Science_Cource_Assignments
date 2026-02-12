WITH citation_velocity AS (
    SELECT
        UserID,
        Country_Origin,
        Year,
        Research_Citations,
        AVG(Research_Citations) OVER (
            PARTITION BY Country_Origin
            ORDER BY Year
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS moving_avg_citations
    FROM Professionals_Data
)
SELECT
    UserID,
    Country_Origin,
    Year,
    Research_Citations,
    moving_avg_citations,
    DENSE_RANK() OVER (
        PARTITION BY Country_Origin
        ORDER BY moving_avg_citations DESC
    ) AS country_rank
FROM citation_velocity
ORDER BY Country_Origin, country_rank, Year;
