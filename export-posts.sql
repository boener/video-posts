-- Export blog posts with URLs for vertical-posts pipeline
-- Paste this into phpMyAdmin's SQL tab and run it, then export the result as JSON.
--
-- Change the two numbers on the BETWEEN line to select a different range of posts.
-- Or remove that line entirely to export all active posts.

SELECT
    bp.post_id,
    bp.post_title,
    bp.opening_paragraph,
    bp.post_content,
    bp.meta_title,
    bp.meta_description,
    bp.meta_keywords,
    bp.post_date,
    bp.post_image,
    u.url
FROM blog_posts AS bp
LEFT OUTER JOIN (
    SELECT associated_id, url
    FROM urls
    WHERE type = '4'
    GROUP BY associated_id
) AS u ON bp.post_id = u.associated_id
WHERE bp.post_active = 1
AND bp.post_id BETWEEN 609 AND 687
ORDER BY bp.post_date DESC;
