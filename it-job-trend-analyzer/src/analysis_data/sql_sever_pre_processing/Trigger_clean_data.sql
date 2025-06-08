-- Trigger để xóa liên đới từ JOB_DETAIL khi xóa JOB
CREATE TRIGGER trg_DeleteJobWithDetails
ON JOB
INSTEAD OF DELETE
AS
BEGIN
    -- Xoá trước các dòng liên quan trong JOB_DETAIL
    DELETE FROM JOB_DETAIL
    WHERE JOB_ID IN (SELECT JOB_ID FROM DELETED);

    -- Sau đó xoá chính dòng trong JOB
    DELETE FROM JOB
    WHERE JOB_ID IN (SELECT JOB_ID FROM DELETED);
END;


